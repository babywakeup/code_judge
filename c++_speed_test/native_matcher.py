"""Python-facing API: validate arrays, then delegate only the hot loop to C++."""
import ctypes
import numpy as np
from build import LIBRARY

_library = None


def prepare(image, template):
    arrays = []
    for value in (image, template):
        array = np.asarray(value)
        if array.ndim != 2 or array.dtype != np.uint8 or not array.size:
            raise ValueError("Expected nonempty 2D uint8 grayscale images")
        arrays.append(np.ascontiguousarray(array))
    image, template = arrays
    if any(t > i for t, i in zip(template.shape, image.shape)):
        raise ValueError("Template must fit inside the image")
    return image, template


def match_cpp(image, template):
    global _library
    image, template = prepare(image, template)
    if _library is None:
        if not LIBRARY.exists():
            raise RuntimeError("Run python build.py first to compile the C++ library")
        _library = ctypes.CDLL(str(LIBRARY))
        array_type = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS")
        _library.match_ssd.argtypes = [array_type, ctypes.c_int, ctypes.c_int,
                                      array_type, ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_double)]
        _library.match_ssd.restype = ctypes.c_int
    x, y, score = ctypes.c_int(), ctypes.c_int(), ctypes.c_double()
    status = _library.match_ssd(image, *image.shape, template, *template.shape,
                               ctypes.byref(x), ctypes.byref(y), ctypes.byref(score))
    if status:
        raise ValueError(f"C++ matching failed: status={status}")
    return x.value, y.value, score.value


def match_python(image, template):
    """Same algorithm and tie ordering; intentionally uses Python pixel loops."""
    image, template = prepare(image, template)
    h, w = image.shape
    th, tw = template.shape
    best = (0, 0, float("inf"))
    for y in range(h - th + 1):
        for x in range(w - tw + 1):
            score = 0
            for ty in range(th):
                for tx in range(tw):
                    diff = int(image[y + ty, x + tx]) - int(template[ty, tx])
                    score += diff * diff
            if score < best[2]:
                best = (x, y, score)
    return best
