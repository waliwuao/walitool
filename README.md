# Walitool

**A high-performance scientific computing library for Python, implemented in Rust.**

`walitool` is designed to provide fast and efficient implementations of common Multi-Criteria Decision Making (MCDM) algorithms. By leveraging Rust's `ndarray` and `PyO3`, it offers significant performance improvements over pure Python implementations, while maintaining seamless compatibility with the NumPy ecosystem.

## Features

- **🚀 High Performance**: Core algorithms are written in Rust, ensuring maximum speed and memory efficiency.
- **⚡ NumPy Compatible**: Accepts standard NumPy arrays as input and returns NumPy arrays.
- **🧮 Algorithms**:
  - **Entropy Weight Method (EWM)**: Objective weighting method based on information entropy.
  - **TOPSIS**: Technique for Order of Preference by Similarity to Ideal Solution.

## Installation

You can install the library directly from PyPI:

```bash
pip install walitool
```

## Quick Start

Here is a complete example of how to evaluate 5 different policies using **Entropy Weight Method** combined with **TOPSIS**.

### Prerequisite

```bash
pip install numpy pandas
```

### Complete Workflow Example

```python
import numpy as np
import pandas as pd
import walitool

# 1. Prepare Data
# Rows: 5 Policies (P1 to P5)
# Cols: 3 Indicators (Cost, Efficiency, Sustainability)
data = np.array([
    [100, 90, 80],  # P1
    [80,  70, 60],  # P2
    [90,  85, 88],  # P3
    [120, 95, 70],  # P4
    [85,  75, 95]   # P5
], dtype=float)

# 2. Define Indicator Types
# 1 = Positive Indicator (The higher, the better), e.g., Efficiency
# 0 = Negative Indicator (The lower, the better), e.g., Cost
# Order: [Cost, Efficiency, Sustainability] -> [Negative, Positive, Positive]
types = np.array([0, 1, 1], dtype=np.int64)

print("--- Step 1: Calculate Weights using Entropy Method ---")
# Returns: (weights, internal_scores)
weights, _ = walitool.entropy_weight(data, types)

# Format output with Pandas for readability
df_weights = pd.DataFrame(weights, index=["Cost", "Efficiency", "Sustainability"], columns=["Weight"])
print(df_weights.round(4))

print("\n--- Step 2: Calculate Final Scores using TOPSIS ---")
# Input: Data, Calculated Weights, and Indicator Types
topsis_scores = walitool.topsis(data, weights, types)

# Rank the policies
df_scores = pd.DataFrame({
    "Policy": ["P1", "P2", "P3", "P4", "P5"],
    "Score": topsis_scores
})
df_scores["Rank"] = df_scores["Score"].rank(ascending=False).astype(int)
df_scores = df_scores.sort_values("Rank")

print(df_scores.round(4))
```

## API Reference

### 1. `entropy_weight(data, indicator_type)`

Calculates weights using the Information Entropy method.

- **Parameters**:
    - `data` (`numpy.ndarray[float64]`): A 2D array where rows represent samples and columns represent indicators.
    - `indicator_type` (`numpy.ndarray[int64]`): A 1D array indicating the type of each column. `1` for positive (benefit) type, `0` for negative (cost) type.
- **Returns**:
    - `tuple(weights, scores)`:
        - `weights`: A 1D array of calculated weights for each indicator.
        - `scores`: A 1D array of comprehensive scores based on EWM (simple weighted sum).

### 2. `topsis(data, weights, indicator_type)`

Calculates the relative closeness to the ideal solution using the TOPSIS method.

- **Parameters**:
    - `data` (`numpy.ndarray[float64]`): A 2D array where rows represent samples and columns represent indicators.
    - `weights` (`numpy.ndarray[float64]`): A 1D array of weights corresponding to each indicator (must sum to 1 ideally, though the algorithm handles normalization).
    - `indicator_type` (`numpy.ndarray[int64]`): A 1D array indicating the type of each column (`1` or `0`).
- **Returns**:
    - `scores` (`numpy.ndarray[float64]`): A 1D array of TOPSIS scores (0 to 1), where a higher value indicates a better solution.

## Performance Note

Since `walitool` compiles to native machine code, it is significantly faster than equivalent pure Python implementations, especially when processing large datasets or performing batch calculations.

## License

This project is licensed under the MIT License.
