# Walitool

**A high-performance scientific computing library for Python, implemented in Rust.**

`walitool` is designed to provide fast and efficient implementations of Multi-Criteria Decision Making (MCDM) algorithms and heuristic optimization methods. By leveraging Rust's `ndarray` and `PyO3`, it delivers significant performance improvements over pure Python implementations while maintaining seamless compatibility with the NumPy ecosystem.

## Features

- **🚀 Native Performance**: Core algorithms are written in Rust, ensuring maximum execution speed and memory efficiency.
- **⚡ NumPy Integration**: Accepts standard NumPy arrays as input and returns NumPy arrays.
- **🧬 Optimization Engine**: High-performance Differential Evolution (DE) supporting Numba-accelerated callbacks to eliminate cross-language overhead.
- **🧮 Integrated Algorithms**:
  - **Entropy Weight Method (EWM)**: Objective weighting based on information entropy.
  - **TOPSIS**: Technique for Order of Preference by Similarity to Ideal Solution.
  - **Differential Evolution (DE)**: Parallelized global optimization for large-scale problems.

## Installation

Install the library directly from PyPI:

```bash
pip install walitool
```

## Quick Start

### 1. Multi-Criteria Decision Making (EWM + TOPSIS)

Evaluate 5 different policies based on 3 indicators (Cost, Efficiency, Sustainability).

```python
import numpy as np
import walitool

# Data: 5 Policies, 3 Indicators
data = np.array([
    [100, 90, 80], [80, 70, 60], [90, 85, 88], [120, 95, 70], [85, 75, 95]
], dtype=float)

# Indicator Types: 1 = Positive (Higher is better), 0 = Negative (Lower is better)
# Order: [Cost, Efficiency, Sustainability] -> [Negative, Positive, Positive]
types = np.array([0, 1, 1], dtype=np.int64)

# Step 1: Calculate Weights using Entropy Weight Method
weights, _ = walitool.entropy_weight(data, types)

# Step 2: Calculate Comprehensive Scores using TOPSIS
scores = walitool.topsis(data, weights, types)

print(f"Calculated Weights: {weights}")
print(f"Final Scores: {scores}")
```

### 2. Differential Evolution (DE) Optimization

`walitool` supports **Native Callback Mode** via Numba, allowing Rust to call your Python objective function at machine-code speed.

```python
import numpy as np
from walitool import DE

# 1. Define a vectorized objective function (e.g., Rastrigin)
def objective_function(x):
    # x has shape (pop_size, dimensions)
    A = 10
    return A * x.shape[1] + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)

# 2. Configure Optimizer
dim = 50
bounds = (np.full(dim, -5.12), np.full(dim, 5.12))
options = {'F': 0.5, 'CR': 0.9}

# 3. Instantiate and Optimize
optimizer = DE(pop=100, dimensions=dim, bounds=bounds, options=options)
best_cost, best_pos = optimizer.optimize(objective_function, iters=20000)

print(f"Global Minimum Cost: {best_cost:.4e}")
```

## API Reference

### 1. `entropy_weight(data, indicator_type)`
Calculates weights using the Information Entropy method.
- **Parameters**:
    - `data` (`ndarray`): 2D array (samples × indicators).
    - `indicator_type` (`ndarray`): 1D array (`1` for benefit, `0` for cost).
- **Returns**: `(weights, scores)` tuple.

### 2. `topsis(data, weights, indicator_type)`
Calculates relative closeness to the ideal solution.
- **Parameters**:
    - `weights`: Indicator weights (e.g., from `entropy_weight`).
- **Returns**: 1D array of scores (0 to 1).

### 3. `DE(pop, dimensions, bounds, options)`
Differential Evolution optimizer.
- **Parameters**:
    - `pop`: Population size (recommended 5–10 × dimensions).
    - `dimensions`: Problem dimensionality.
    - `bounds`: Tuple of `(min_array, max_array)` defining search space.
    - `options`: Dictionary containing mutation factor `F` and crossover probability `CR`.
- **Method**:
    - `optimize(fitness_function, iters)`: Performs optimization. The `fitness_function` should be vectorized (receive `(N, D)` and return `(N,)`).

## Performance Insights

- **Numba Acceleration**: When using `DE`, installing `numba` is highly recommended. `walitool` automatically compiles your Python function and passes a raw C function pointer to the Rust core, bypassing the Python interpreter's loop overhead.
- **Parallel Mutation**: The Rust core utilizes the `Rayon` library for parallel population mutation, significantly speeding up high-dimensional or large-population tasks on multi-core CPUs.
- **FFI Optimization**: Uses zero-copy memory mapping for data transfer between Python and Rust.

## License

This project is licensed under the MIT License.