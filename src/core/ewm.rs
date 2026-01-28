use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

pub fn calculate_ewm(
    data: ArrayView2<f64>,
    indicator_types: ArrayView1<i64>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let (rows, cols) = data.dim();
    if rows <= 1 { return Err("Data must have more than one row".to_string()); }

    let data_t = data.t().to_owned();
    let mut entropies = Array1::<f64>::zeros(cols);
    let mut normalized_t = unsafe { Array2::<f64>::uninit(data_t.dim()).assume_init() };

    Zip::from(normalized_t.rows_mut())
        .and(data_t.rows())
        .and(&indicator_types)
        .and(&mut entropies)
        .par_for_each(|mut norm_row, in_row, &ind_type, e| {
            let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
            for &x in in_row {
                if x < min { min = x; }
                if x > max { max = x; }
            }

            let range = max - min;
            if range == 0.0 {
                norm_row.fill(1.0);
                *e = 1.0;
            } else {
                let mut sum_v = 0.0;
                let mut sum_v_ln_v = 0.0;
                
                if ind_type == 1 {
                    for (&x, n) in in_row.iter().zip(norm_row.iter_mut()) {
                        let v = (x - min) / range;
                        *n = v;
                        sum_v += v;
                        if v > 1e-12 { sum_v_ln_v += v * v.ln(); }
                    }
                } else {
                    for (&x, n) in in_row.iter().zip(norm_row.iter_mut()) {
                        let v = (max - x) / range;
                        *n = v;
                        sum_v += v;
                        if v > 1e-12 { sum_v_ln_v += v * v.ln(); }
                    }
                }

                if sum_v > 0.0 {
                    let k = 1.0 / (rows as f64).ln();
                    let val = (sum_v_ln_v / sum_v) - sum_v.ln();
                    *e = -k * val;
                } else {
                    *e = 1.0;
                }
            }
        });

    let mut weights = 1.0 - &entropies;
    let div_sum = weights.sum();
    if div_sum == 0.0 {
        weights.fill(1.0 / cols as f64);
    } else {
        weights /= div_sum;
    }

    let scores = normalized_t.t().dot(&weights);
    Ok((weights, scores))
}