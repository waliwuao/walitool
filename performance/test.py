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
best_cost, best_pos = optimizer.optimize(objective_function, iters=50000)

print(f"Global Minimum Cost: {best_cost:.4e}")