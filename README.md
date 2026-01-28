# Walitool

A high-performance scientific computing library for Python, implemented in Rust.

## Installation

```bash
pip install walitool
```

## Usage

```python
import numpy as np
import walitool

# Data: 5 rows (policies), 6 columns (indicators)
data = np.random.rand(5, 6)
# Indicator types: 1 for positive, 0 for negative
types = np.array([1, 0, 1, 0, 1, 0], dtype=np.int64)

weights, scores = walitool.entropy_weight(data, types)
print(weights)
print(scores)
```

## License

MIT
