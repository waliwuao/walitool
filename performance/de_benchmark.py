import numpy as np
import walitool
import time
import statistics

def rastrigin(x):
    A = 10
    d = x.shape[1]
    return A * d + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)

class NumpyDE:
    def __init__(self, pop_size, dimensions, bounds, options):
        self.pop_size = pop_size
        self.dim = dimensions
        self.lb = bounds[0]
        self.ub = bounds[1]
        self.F = options.get('F', 0.5)
        self.CR = options.get('CR', 0.7)

    def optimize(self, func, iters):
        pop = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
        fitness = func(pop)
        best_idx = np.argmin(fitness)
        best_cost = fitness[best_idx]
        best_pos = pop[best_idx].copy()
        for _ in range(iters):
            idxs = np.random.randint(0, self.pop_size, size=(self.pop_size, 3))
            a, b, c = pop[idxs[:, 0]], pop[idxs[:, 1]], pop[idxs[:, 2]]
            mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)
            cross_points = np.random.rand(self.pop_size, self.dim) < self.CR
            trial = np.where(cross_points, mutant, pop)
            f_trial = func(trial)
            improved = f_trial < fitness
            pop[improved] = trial[improved]
            fitness[improved] = f_trial[improved]
            min_val = np.min(fitness)
            if min_val < best_cost:
                best_cost = min_val
                best_pos = pop[np.argmin(fitness)].copy()
        return best_cost, best_pos

def run_benchmark(name, optimizer_cls, func, iters, runs=3, **kwargs):
    times = []
    final_costs = []
    if name == "Rust(Walitool)":
        warmup_opt = optimizer_cls(pop=10, dimensions=2, options=kwargs['options'], bounds=(np.array([-5,-5]), np.array([5,5])))
        warmup_opt.optimize(func, iters=1)
    print(f"Testing {name:^15} | Scale: D={kwargs['dimensions']}, N={kwargs['pop']}, G={iters} ", end="", flush=True)
    for _ in range(runs):
        if name == "Rust(Walitool)":
            opt = optimizer_cls(pop=kwargs['pop'], dimensions=kwargs['dimensions'], options=kwargs['options'], bounds=kwargs['bounds'])
        else:
            opt = optimizer_cls(pop_size=kwargs['pop'], dimensions=kwargs['dimensions'], bounds=kwargs['bounds'], options=kwargs['options'])
        start = time.perf_counter()
        cost, _ = opt.optimize(func, iters=iters)
        end = time.perf_counter()
        times.append(end - start)
        final_costs.append(cost)
        print(".", end="", flush=True)
    avg_time = statistics.mean(times)
    avg_cost = statistics.mean(final_costs)
    print(f" Done. Avg: {avg_time:.3f}s")
    return avg_time, avg_cost

if __name__ == "__main__":
    scenarios = [
        {"name": "基准大规模 (D500, G2000)", "dim": 500, "pop": 200, "iters": 2000},
        {"name": "超长周期 (D100, G20000)", "dim": 100, "pop": 100, "iters": 20000}
    ]
    options = {'F': 0.5, 'CR': 0.9}
    print("="*100)
    print(f"{'Walitool 大规模演化计算性能报告':^100}")
    print("="*100)
    print(f"{'Scenario':<30} | {'Engine':<15} | {'Time (s)':<12} | {'Speedup':<8} | {'Min Cost':<10}")
    print("-" * 100)
    for sc in scenarios:
        dim, pop, iters = sc['dim'], sc['pop'], sc['iters']
        bounds = (np.full(dim, -5.12), np.full(dim, 5.12))
        py_time, py_cost = run_benchmark("Python(NumPy)", NumpyDE, rastrigin, iters, pop=pop, dimensions=dim, bounds=bounds, options=options)
        rs_time, rs_cost = run_benchmark("Rust(Walitool)", walitool.DE, rastrigin, iters, pop=pop, dimensions=dim, bounds=bounds, options=options)
        speedup = py_time / rs_time
        print(f"{sc['name']:<30} | {'Py(NumPy)':<15} | {py_time:.4f} | {'1.0x':<8} | {py_cost:.2e}")
        print(f"{' ': <30} | {'Rust(Wali)':<15} | {rs_time:.4f} | {speedup:.2f}x | {rs_cost:.2e}")
        print("-" * 100)