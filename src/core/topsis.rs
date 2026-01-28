use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

pub fn calculate_topsis(
    data: ArrayView2<f64>,
    weights: ArrayView1<f64>,
    indicator_types: ArrayView1<i64>,
) -> Result<Array1<f64>, String> {
    let (rows, cols) = data.dim();

    if rows <= 1 {
        return Err("Data must have more than one row".to_string());
    }
    if cols != weights.len() {
        return Err("Number of columns must match number of weights".to_string());
    }
    if cols != indicator_types.len() {
        return Err("Number of columns must match number of indicator types".to_string());
    }

    let mut weighted_normalized = Array2::<f64>::zeros((rows, cols));

    for j in 0..cols {
        let col = data.index_axis(Axis(1), j);
        let sum_sq = col.fold(0.0, |acc, &x| acc + x * x);
        let norm_factor = sum_sq.sqrt();

        let w = weights[j];
        
        for i in 0..rows {
            if norm_factor == 0.0 {
                weighted_normalized[[i, j]] = 0.0;
            } else {
                weighted_normalized[[i, j]] = (col[i] / norm_factor) * w;
            }
        }
    }

    let mut ideal_best = Array1::<f64>::zeros(cols);
    let mut ideal_worst = Array1::<f64>::zeros(cols);

    for j in 0..cols {
        let col = weighted_normalized.index_axis(Axis(1), j);
        let max_val = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = col.fold(f64::INFINITY, |a, &b| a.min(b));

        if indicator_types[j] == 1 {
            ideal_best[j] = max_val;
            ideal_worst[j] = min_val;
        } else {
            ideal_best[j] = min_val;
            ideal_worst[j] = max_val;
        }
    }

    let mut scores = Array1::<f64>::zeros(rows);

    for i in 0..rows {
        let row = weighted_normalized.index_axis(Axis(0), i);
        
        let mut dist_best_sq_sum = 0.0;
        let mut dist_worst_sq_sum = 0.0;

        for j in 0..cols {
            let diff_best = row[j] - ideal_best[j];
            let diff_worst = row[j] - ideal_worst[j];
            dist_best_sq_sum += diff_best * diff_best;
            dist_worst_sq_sum += diff_worst * diff_worst;
        }

        let dist_best = dist_best_sq_sum.sqrt();
        let dist_worst = dist_worst_sq_sum.sqrt();

        if dist_best + dist_worst == 0.0 {
            scores[i] = 0.5;
        } else {
            scores[i] = dist_worst / (dist_best + dist_worst);
        }
    }

    Ok(scores)
}