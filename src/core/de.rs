use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::exceptions::{PyValueError, PyTypeError};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, ToPyArray};
use ndarray::{Array1, Array2, Zip, ArrayView2, ArrayViewMut1, ArrayView1};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::Uniform;

type FitnessFnC = unsafe extern "C" fn(n: usize, d: usize, input: *const f64, output: *mut f64);

#[pyclass(subclass)]
pub struct _DE {
    pop_size: usize,
    dimensions: usize,
    bounds_min: Array1<f64>,
    bounds_max: Array1<f64>,
    f_weight: f64,
    cr_prob: f64,
}

#[pymethods]
impl _DE {
    #[new]
    #[pyo3(signature = (pop, dimensions, options, bounds))]
    fn new(
        pop: usize,
        dimensions: usize,
        options: &Bound<'_, PyDict>,
        bounds: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        if pop < 4 {
            return Err(PyValueError::new_err("Population size must be at least 4 for DE"));
        }

        let f_weight = if let Some(val) = options.get_item("F")? {
            val.extract::<f64>().unwrap_or(0.5)
        } else if let Some(val) = options.get_item("c1")? {
            val.extract::<f64>().unwrap_or(0.5)
        } else {
            0.5
        };

        let cr_prob = if let Some(val) = options.get_item("CR")? {
            val.extract::<f64>().unwrap_or(0.7)
        } else if let Some(val) = options.get_item("c2")? {
            val.extract::<f64>().unwrap_or(0.7)
        } else {
            0.7
        };

        if bounds.len() != 2 {
            return Err(PyValueError::new_err("Bounds must be a tuple of (min_bound, max_bound)"));
        }

        let min_bound_py: PyReadonlyArray1<f64> = bounds.get_item(0)?.extract()?;
        let max_bound_py: PyReadonlyArray1<f64> = bounds.get_item(1)?.extract()?;

        let bounds_min = min_bound_py.as_array().to_owned();
        let bounds_max = max_bound_py.as_array().to_owned();

        if bounds_min.len() != dimensions || bounds_max.len() != dimensions {
            return Err(PyValueError::new_err("Bounds dimensions must match problem dimensions"));
        }

        Ok(_DE {
            pop_size: pop,
            dimensions,
            bounds_min,
            bounds_max,
            f_weight,
            cr_prob,
        })
    }

    fn optimize<'py>(
        &self,
        py: Python<'py>,
        fitness_function: Bound<'py, PyAny>,
        iters: usize,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        if let Ok(ptr_val) = fitness_function.extract::<usize>() {
            let func_ptr: FitnessFnC = unsafe { std::mem::transmute(ptr_val) };
            
            let (best_cost, best_pos_vec) = py.allow_threads(move || {
                self.run_native_optimization(func_ptr, iters)
            });
            
            let best_pos_arr = Array1::from_vec(best_pos_vec);
            return Ok((best_cost, best_pos_arr.into_pyarray(py)));
        }
        
        if fitness_function.is_callable() {
            return self.run_python_optimization(py, fitness_function, iters);
        }

        Err(PyTypeError::new_err("Fitness function must be a callable or an address int"))
    }
}

impl _DE {
    fn run_native_optimization(&self, fitness_fn: FitnessFnC, iters: usize) -> (f64, Vec<f64>) {
        let mut population = Array2::<f64>::zeros((self.pop_size, self.dimensions));
        let mut rng = SmallRng::from_entropy();
        let range_dist = Uniform::new(0.0, 1.0);
        
        for mut row in population.rows_mut() {
            for (i, x) in row.iter_mut().enumerate() {
                unsafe {
                    let min = *self.bounds_min.uget(i);
                    let max = *self.bounds_max.uget(i);
                    *x = min + range_dist.sample(&mut rng) * (max - min);
                }
            }
        }
        
        let mut costs = Array1::<f64>::zeros(self.pop_size);
        unsafe { fitness_fn(self.pop_size, self.dimensions, population.as_ptr(), costs.as_mut_ptr()); }

        let mut best_cost = f64::INFINITY;
        let mut best_pos = Array1::<f64>::zeros(self.dimensions);

        for (i, &c) in costs.iter().enumerate() {
            if c < best_cost {
                best_cost = c;
                best_pos.assign(&population.row(i));
            }
        }

        let mut trial_pop = Array2::<f64>::zeros((self.pop_size, self.dimensions));
        let mut trial_costs = Array1::<f64>::zeros(self.pop_size);
        
        let use_parallel = (self.pop_size * self.dimensions) > 200_000;

        for _ in 0..iters {
            let pop_view = population.view();
            if use_parallel {
                Zip::from(trial_pop.rows_mut()).and(population.rows()).par_for_each(|trial_row, target_row| {
                    let mut local_rng = SmallRng::seed_from_u64(rand::random());
                    Self::mutate_row_unsafe(trial_row, target_row, &pop_view, &self.bounds_min, &self.bounds_max, self.pop_size, self.dimensions, self.f_weight, self.cr_prob, &mut local_rng);
                });
            } else {
                Zip::from(trial_pop.rows_mut()).and(population.rows()).for_each(|trial_row, target_row| {
                    Self::mutate_row_unsafe(trial_row, target_row, &pop_view, &self.bounds_min, &self.bounds_max, self.pop_size, self.dimensions, self.f_weight, self.cr_prob, &mut rng);
                });
            }

            unsafe { fitness_fn(self.pop_size, self.dimensions, trial_pop.as_ptr(), trial_costs.as_mut_ptr()); }

            for i in 0..self.pop_size {
                let tc = unsafe { *trial_costs.uget(i) };
                let current_cost = unsafe { *costs.uget(i) };
                if tc < current_cost {
                    unsafe { *costs.uget_mut(i) = tc; }
                    let source = trial_pop.row(i);
                    let mut dest = population.row_mut(i);
                    dest.assign(&source);
                    if tc < best_cost {
                        best_cost = tc;
                        best_pos.assign(&dest);
                    }
                }
            }
        }
        (best_cost, best_pos.to_vec())
    }

    fn run_python_optimization<'py>(&self, py: Python<'py>, fitness_function: Bound<'py, PyAny>, iters: usize) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let mut population = Array2::<f64>::zeros((self.pop_size, self.dimensions));
        let mut rng = SmallRng::from_entropy();
        let range_dist = Uniform::new(0.0, 1.0);
        
        for mut row in population.rows_mut() {
            for (i, x) in row.iter_mut().enumerate() {
                unsafe {
                    let min = *self.bounds_min.uget(i);
                    let max = *self.bounds_max.uget(i);
                    *x = min + range_dist.sample(&mut rng) * (max - min);
                }
            }
        }

        let py_pop = population.to_pyarray(py); 
        let cost_result = fitness_function.call1((py_pop,))?;
        let cost_array: PyReadonlyArray1<f64> = cost_result.extract()?;
        let mut costs = cost_array.as_array().to_owned();

        let mut best_cost = f64::INFINITY;
        let mut best_pos = Array1::<f64>::zeros(self.dimensions);

        for (i, &c) in costs.iter().enumerate() {
            if c < best_cost {
                best_cost = c;
                best_pos.assign(&population.row(i));
            }
        }

        let mut trial_pop = Array2::<f64>::zeros((self.pop_size, self.dimensions));
        
        for _ in 0..iters {
            let pop_view = population.view();
            
            Zip::from(trial_pop.rows_mut())
                .and(population.rows())
                .for_each(|trial_row, target_row| {
                    Self::mutate_row_unsafe(
                        trial_row, target_row, &pop_view, 
                        &self.bounds_min, &self.bounds_max, 
                        self.pop_size, self.dimensions, 
                        self.f_weight, self.cr_prob, &mut rng
                    );
                });

            let py_trial = trial_pop.to_pyarray(py);
            let trial_result = fitness_function.call1((py_trial,))?;
            let trial_cost_obj: PyReadonlyArray1<f64> = trial_result.extract()?;
            let trial_costs = trial_cost_obj.as_array();

            if trial_costs.len() != self.pop_size {
                return Err(PyValueError::new_err("Fitness function returned incorrect shape"));
            }

            for i in 0..self.pop_size {
                let tc = unsafe { *trial_costs.uget(i) };
                let current_cost = unsafe { *costs.uget(i) };
                
                if tc < current_cost {
                    unsafe { *costs.uget_mut(i) = tc };
                    let source = trial_pop.row(i);
                    let mut dest = population.row_mut(i);
                    dest.assign(&source);

                    if tc < best_cost {
                        best_cost = tc;
                        best_pos.assign(&dest);
                    }
                }
            }
        }

        Ok((best_cost, best_pos.into_pyarray(py)))
    }

    #[inline(always)]
    fn mutate_row_unsafe(
        mut trial_row: ArrayViewMut1<f64>,
        target_row: ArrayView1<f64>,
        pop_view: &ArrayView2<f64>,
        bounds_min: &Array1<f64>,
        bounds_max: &Array1<f64>,
        pop_size: usize,
        dimensions: usize,
        f_weight: f64,
        cr_prob: f64,
        rng: &mut impl Rng,
    ) {
        let a = rng.gen_range(0..pop_size);
        let mut b = rng.gen_range(0..pop_size);
        while b == a { b = rng.gen_range(0..pop_size); }
        let mut c = rng.gen_range(0..pop_size);
        while c == a || c == b { c = rng.gen_range(0..pop_size); }

        let idx_r = rng.gen_range(0..dimensions);

        for j in 0..dimensions {
            let should_crossover = rng.gen::<f64>() < cr_prob || j == idx_r;
            unsafe {
                if should_crossover {
                    let val_a = *pop_view.uget((a, j));
                    let val_b = *pop_view.uget((b, j));
                    let val_c = *pop_view.uget((c, j));
                    let val = val_a + f_weight * (val_b - val_c);
                    
                    let min_v = *bounds_min.uget(j);
                    let max_v = *bounds_max.uget(j);
                    
                    if val < min_v { *trial_row.uget_mut(j) = min_v; }
                    else if val > max_v { *trial_row.uget_mut(j) = max_v; }
                    else { *trial_row.uget_mut(j) = val; }
                } else {
                    *trial_row.uget_mut(j) = *target_row.uget(j);
                }
            }
        }
    }
}