from __future__ import annotations

import ctypes
import os
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOFILTERPY_LIB") or os.path.join(
    ROOT, "dist", "libmojo-filterpy.so"
)
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mfp_predict": ([I] * 10 + [F], None),
    "mfp_linear_update": ([I] * 11, I),
    "mfp_unscented_transform": ([I] * 9, None),
    "mfp_unscented_transform_gpu": ([I] * 9, I),
    "mfp_cross_variance": ([I] * 9, None),
    "mfp_ukf_update": ([I] * 10, I),
    "mfp_resample_sorted": ([I] * 4, None),
    "mfp_resample_binary": ([I] * 5, None),
    "mfp_neff": ([I, I], F),
}

_F64 = np.dtype(np.float64)
_I64 = np.dtype(np.int64)
_NATIVE_DTYPES = frozenset((_F64, _I64))
_POOL = threading.local()

_library = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError(f"Mojo library not found at {LIB}; run `pixi run build`")
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def scratch(count: int, dtype=np.float64) -> tuple[np.ndarray, int]:
    """Thread-local scratch buffer together with its cached address."""
    pool = getattr(_POOL, "buffers", None)
    if pool is None:
        pool = {}
        _POOL.buffers = pool
    key = (count, dtype)
    entry = pool.get(key)
    if entry is None:
        buffer = np.empty(count, dtype=dtype)
        entry = (buffer, int(buffer.ctypes.data))
        pool[key] = entry
    return entry


def bucket_count(n: int) -> int:
    """Search-table size: one bucket per entry, capped so the table stays cacheable."""
    return max(min(n, 1 << 20), 1)


def f64(value, *, copy: bool = False) -> np.ndarray:
    if type(value) is np.ndarray and value.dtype is _F64:
        if not value.flags.c_contiguous:
            value = np.ascontiguousarray(value)
        return value.copy() if copy else value
    source = np.asarray(value)
    if source.dtype.kind not in "bifu":
        raise TypeError(f"expected real numeric data, got dtype {source.dtype}")
    if source.dtype.kind == "f" and source.dtype.itemsize > 8:
        raise TypeError(f"refusing to narrow {source.dtype} to float64")
    if source.dtype.kind in "iu" and source.size:
        converted = source.astype(np.float64)
        if not np.array_equal(converted.astype(source.dtype), source):
            raise ValueError("integer values cannot be represented exactly as float64")
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def i64(value) -> np.ndarray:
    if type(value) is np.ndarray and value.dtype is _I64 and value.flags.c_contiguous:
        return value
    return np.ascontiguousarray(value, dtype=np.int64)


def _data_pointer(array: np.ndarray) -> int:
    try:
        return ctypes.addressof(ctypes.c_char.from_buffer(array))
    except TypeError:
        return int(array.ctypes.data)


def _reject(array) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError("native buffers must be non-empty")
    if not array.flags.c_contiguous:
        raise ValueError("native buffers must be C-contiguous")
    raise TypeError("native buffers must have dtype float64 or int64")


def addr(array: np.ndarray) -> int:
    if (
        type(array) is np.ndarray
        and array.flags.c_contiguous
        and array.dtype in _NATIVE_DTYPES
        and array.size != 0
    ):
        return _data_pointer(array)
    _reject(array)
