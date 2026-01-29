import numpy as np
import warnings
from ._walitool import _DE, entropy_weight, topsis

try:
    from numba import cfunc, types, carray, njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

_NUMBA_CACHE = {}

class DE(_DE):
    def __new__(cls, pop=20, dimensions=2, options=None, bounds=None):
        if options is None:
            options = {'F': 0.5, 'CR': 0.7}
        
        if bounds is not None:
            lb = np.asarray(bounds[0], dtype=np.float64)
            ub = np.asarray(bounds[1], dtype=np.float64)
            bounds = (lb, ub)
            
        return super().__new__(cls, pop, dimensions, options, bounds)

    def __init__(self, pop=20, dimensions=2, options=None, bounds=None):
        pass

    def optimize(self, fitness_function, iters=100):
        if not HAS_NUMBA:
            warnings.warn("Numba not installed. Running in Slow Mode (Python callback).")
            return super().optimize(fitness_function, iters)

        try:
            native_addr = self._compile_to_native(fitness_function)
            return super().optimize(native_addr, iters)
        except Exception:
            return super().optimize(fitness_function, iters)

    def _compile_to_native(self, py_func):
        if py_func in _NUMBA_CACHE:
            return _NUMBA_CACHE[py_func]

        jitted_func = njit(fastmath=True)(py_func)
        
        sig = types.void(
            types.uint64, types.uint64, types.CPointer(types.float64), types.CPointer(types.float64)
        )

        @cfunc(sig)
        def wrapper(n, d, in_ptr, out_ptr):
            input_arr = carray(in_ptr, (n, d), dtype=np.float64)
            output_arr = carray(out_ptr, (n,), dtype=np.float64)
            
            output_arr[:] = jitted_func(input_arr)
        
        addr = wrapper.address
        _NUMBA_CACHE[py_func] = addr
        return addr