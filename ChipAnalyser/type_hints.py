from typing import Literal
import numpy as np

ColorImage = np.ndarray[tuple[int, int, Literal[3]], np.uint8]
GrayImage = np.ndarray[tuple[int, int], np.uint8]
Image = ColorImage | GrayImage

OpenCVIntArray = np.ndarray[tuple[int, Literal[1], Literal[2]], np.int32]
OpenCVFloatArray = np.ndarray[tuple[int, Literal[1], Literal[2]], np.float32]

IntPt = tuple[int, int]  # (x, y) integer coordinates
FloatPt = tuple[float, float]  # (x, y) floating-point coordinates
Line = tuple[float, float, float]  # (rho, xn, yn), xn = cos(theta), yn = sin(theta)

IntPtArray = np.ndarray[tuple[int, Literal[2]], int]  # array of (x, y) integer coordinates
FloatPtArray = np.ndarray[tuple[int, Literal[2]], float]  # array of (x, y) floating-points coordinates

IntArray = np.ndarray[tuple[int], int]
FloatArray = np.ndarray[tuple[int], float]
