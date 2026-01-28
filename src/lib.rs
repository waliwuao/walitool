use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};

mod core;
use core::ewm::calculate_ewm;

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
            let py_weights = weights.into_pyarray(py);
            let py_scores = scores.into_pyarray(py);
            Ok((py_weights, py_scores))
        }
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

#[pymodule]
fn walitool(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(entropy_weight, m)?)?;
    Ok(())
}