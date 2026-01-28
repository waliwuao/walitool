use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

pub fn calculate_ewm(
    data: ArrayView2<f64>,
    indicator_types: ArrayView1<i64>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let (rows, cols) = data.dim();

    if rows <= 1 {
        return Err("Data must have more than one row".to_string());
    }
    if cols != indicator_types.len() {
        return Err("Number of columns must match number of indicator types".to_string());
    }

    let mut normalized_data = Array2::<f64>::zeros((rows, cols));

    for j in 0..cols {
        let col = data.index_axis(Axis(1), j);
        let max_val = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = col.fold(f64::INFINITY, |a, &b| a.min(b));
        let range = max_val - min_val;

        let is_positive = indicator_types[j] == 1;

        for i in 0..rows {
            let val = col[i];
            let norm_val = if range == 0.0 {
                1.0
            } else if is_positive {
                (val - min_val) / range
            } else {
                (max_val - val) / range
            };
            normalized_data[[i, j]] = norm_val;
        }
    }

    let col_sums = normalized_data.sum_axis(Axis(0));
    let k = 1.0 / (rows as f64).ln();

    let mut entropies = Array1::<f64>::zeros(cols);

    for j in 0..cols {
        let sum = col_sums[j];
        if sum == 0.0 {
            entropies[j] = 1.0;
            continue;
        }

        let mut p_ln_p_sum = 0.0;
        for i in 0..rows {
            let p = normalized_data[[i, j]] / sum;
            if p > 0.0 {
                p_ln_p_sum += p * p.ln();
            }
        }
        entropies[j] = -k * p_ln_p_sum;
    }

    let divergence = 1.0 - &entropies;
    let div_sum = divergence.sum();

    let weights = if div_sum == 0.0 {
        Array1::from_elem(cols, 1.0 / cols as f64)
    } else {
        divergence / div_sum
    };

    let scores = normalized_data.dot(&weights);

    Ok((weights, scores))
}