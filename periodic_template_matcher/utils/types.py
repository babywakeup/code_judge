"""Array contracts used by the inference API; shapes are documented by callers.

Images: HxW gray or HxWx3/4 BGR/BGRA uint8, in original coordinates.
LabelMap: HxW uint16 class IDs, with 0 representing known background.
Mask: HxW boolean/integer validity or binary class mask, foreground > 0.
Matrix: 3x3 float64 mapping original template coordinates to target coordinates.
"""
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


Array: TypeAlias = NDArray[Any]
Image: TypeAlias = NDArray[np.uint8]
LabelMap: TypeAlias = NDArray[np.uint16]
ClassIds: TypeAlias = NDArray[np.integer[Any]]  # Accepted integer label input.
Mask: TypeAlias = NDArray[np.bool_] | ClassIds
BoolMask: TypeAlias = NDArray[np.bool_]
Matrix: TypeAlias = NDArray[np.float64]
LabelInput: TypeAlias = Mask | Sequence[Mask]
Shape: TypeAlias = tuple[int, int]  # (height, width)
Size: TypeAlias = tuple[int, int]  # OpenCV (width, height)
PathInput: TypeAlias = str | PathLike[str]
Resolver: TypeAlias = Callable[[str], Path]
Diagnostics: TypeAlias = dict[str, Any]  # JSON-safe metrics/report metadata.
