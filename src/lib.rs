use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};

mod core;
use core::ewm::calculate_ewm;
use core::topsis::calculate_topsis;

#[pyfunction]
fn entropy_weight<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    indicator_type: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data_view = data.as_array();
    let type_view = indicator_type.as_array();

    match calculate_ewm(data_view, type_view) {
        Ok((weights, scores)) => {
            Ok((weights.into_pyarray(py), scores.into_pyarray(py)))
        }
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

#[pyfunction]
fn topsis<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    indicator_type: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_view = data.as_array();
    let weights_view = weights.as_array();
    let type_view = indicator_type.as_array();

    match calculate_topsis(data_view, weights_view, type_view) {
        Ok(scores) => Ok(scores.into_pyarray(py)),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

#[pymodule]
fn walitool(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(entropy_weight, m)?)?;
    m.add_function(wrap_pyfunction!(topsis, m)?)?;
    Ok(())
}