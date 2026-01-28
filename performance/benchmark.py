import time
import numpy as np
import walitool
import platform

def numpy_ewm(data, indicator_types):
    rows, cols = data.shape
    normalized_data = np.zeros_like(data)
    
    for j in range(cols):
        col = data[:, j]
        min_val = np.min(col)
        max_val = np.max(col)
        rng = max_val - min_val
        
        if rng == 0:
            normalized_data[:, j] = 1.0
        else:
            if indicator_types[j] == 1:
                normalized_data[:, j] = (col - min_val) / rng
            else:
                normalized_data[:, j] = (max_val - col) / rng

    col_sums = np.sum(normalized_data, axis=0)
    k = 1.0 / np.log(rows)
    entropies = np.zeros(cols)
    
    for j in range(cols):
        s = col_sums[j]
        if s == 0:
            entropies[j] = 1.0
            continue
        p = normalized_data[:, j] / s
        p = p[p > 0]
        entropies[j] = -k * np.sum(p * np.log(p))

    divergence = 1.0 - entropies
    div_sum = np.sum(divergence)
    
    if div_sum == 0:
        return np.ones(cols) / cols
    return divergence / div_sum

def numpy_topsis(data, weights, indicator_types):
    rows, cols = data.shape
    squared_sum = np.sum(data ** 2, axis=0)
    norm_factors = np.sqrt(squared_sum)
    
    weighted_norm = data / norm_factors * weights
    
    ideal_best = np.zeros(cols)
    ideal_worst = np.zeros(cols)
    
    for j in range(cols):
        col = weighted_norm[:, j]
        if indicator_types[j] == 1:
            ideal_best[j] = np.max(col)
            ideal_worst[j] = np.min(col)
        else:
            ideal_best[j] = np.min(col)
            ideal_worst[j] = np.max(col)
            
    dist_best = np.sqrt(np.sum((weighted_norm - ideal_best) ** 2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_norm - ideal_worst) ** 2, axis=1))
    
    scores = dist_worst / (dist_best + dist_worst)
    return scores

def benchmark(rows, cols, iterations=10):
    print(f"\n[Size: {rows} rows x {cols} cols] Running {iterations} iterations...")
    
    data = np.random.rand(rows, cols) * 100
    types = np.random.randint(0, 2, cols).astype(np.int64)
    
    start = time.perf_counter()
    for _ in range(iterations):
        w_py = numpy_ewm(data, types)
        _ = numpy_topsis(data, w_py, types)
    py_time = (time.perf_counter() - start) / iterations

    start = time.perf_counter()
    for _ in range(iterations):
        w_rs, _ = walitool.entropy_weight(data, types)
        _ = walitool.topsis(data, w_rs, types)
    rs_time = (time.perf_counter() - start) / iterations

    print(f"Python (NumPy): {py_time:.6f} s/iter")
    print(f"Rust (Walitool): {rs_time:.6f} s/iter")
    
    if rs_time > 0:
        speedup = py_time / rs_time
        print(f"🚀 Speedup: {speedup:.2f}x")
    else:
        print("🚀 Speedup: Infinite (Rust took ~0s)")

def main():
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print("Benchmark Task: Entropy Weight Method + TOPSIS Pipeline")
    
    test_cases = [
        (1000, 50),
        (10000, 100),
        (50000, 200),
        (100000, 500)
    ]
    
    for r, c in test_cases:
        benchmark(r, c)

if __name__ == "__main__":
    main()