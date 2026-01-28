use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

pub fn calculate_topsis(
    data: ArrayView2<f64>,
    weights: ArrayView1<f64>,
    indicator_types: ArrayView1<i64>,
) -> Result<Array1<f64>, String> {
    let (rows, cols) = data.dim();
    if rows <= 1 { return Err("Data must have more than one row".to_string()); }

    let data_t = data.t().to_owned();
    let mut weighted_norm_t = unsafe { Array2::<f64>::uninit(data_t.dim()).assume_init() };
    let mut ideal_best = Array1::<f64>::zeros(cols);
    let mut ideal_worst = Array1::<f64>::zeros(cols);

    Zip::from(weighted_norm_t.rows_mut())
        .and(data_t.rows())
        .and(&weights)
        .and(&indicator_types)
        .and(&mut ideal_best)
        .and(&mut ideal_worst)
        .par_for_each(|mut out_row, in_row, &w, &ind_type, best, worst| {
            let mut sum_sq = 0.0;
            for &x in in_row { sum_sq += x * x; }
            let norm_factor = sum_sq.sqrt();
            
            let (mut min_v, mut max_v) = (f64::INFINITY, f64::NEG_INFINITY);
            
            if norm_factor != 0.0 {
                let factor = w / norm_factor;
                for (&x, n) in in_row.iter().zip(out_row.iter_mut()) {
                    let v = x * factor;
                    *n = v;
                    if v < min_v { min_v = v; }
                    if v > max_v { max_v = v; }
                }
            } else {
                out_row.fill(0.0);
                min_v = 0.0; max_v = 0.0;
            }

            if ind_type == 1 {
                *best = max_v; *worst = min_v;
            } else {
                *best = min_v; *worst = max_v;
            }
        });

    let mut scores = Array1::<f64>::zeros(rows);
    let weighted_norm = weighted_norm_t.t();

    Zip::from(&mut scores)
        .and(weighted_norm.rows())
        .par_for_each(|s, row| {
            let (mut d_best_sq, mut d_worst_sq) = (0.0, 0.0);
            for j in 0..cols {
                let db = row[j] - ideal_best[j];
                let dw = row[j] - ideal_worst[j];
                d_best_sq += db * db;
                d_worst_sq += dw * dw;
            }
            let d_best = d_best_sq.sqrt();
            let d_worst = d_worst_sq.sqrt();
            *s = if (d_best + d_worst) == 0.0 { 0.5 } else { d_worst / (d_best + d_worst) };
        });

    Ok(scores)
}