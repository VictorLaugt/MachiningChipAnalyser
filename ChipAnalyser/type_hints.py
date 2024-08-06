from typing import Literal, TypeAlias
import numpy as np

ColorImage: TypeAlias = np.ndarray[tuple[int, int, Literal[3]], np.uint8]
GrayImage: TypeAlias = np.ndarray[tuple[int, int], np.uint8]

OpenCVFloatArray: TypeAlias = np.ndarray[tuple[int, Literal[1], Literal[2]], np.float32]

IntPt: TypeAlias = tuple[int, int]  # (x, y) integer coordinates
FloatPt: TypeAlias = tuple[float, float]  # (x, y) floating-point coordinates
Line: TypeAlias = tuple[float, float, float]  # (rho, xn, yn) where xn = cos(theta), yn = sin(theta) and (rho, theta) are the polar parameters of a line

IntPtArray: TypeAlias = np.ndarray[tuple[int, Literal[2]], int]  # array of (x, y) integer coordinates
FloatPtArray: TypeAlias = np.ndarray[tuple[int, Literal[2]], float]  # array of (x, y) floating-points coordinates

IntArray: TypeAlias = np.ndarray[tuple[int], int]  # array of integer scalars
FloatArray: TypeAlias = np.ndarray[tuple[int], float]  # array of floating-point scalars
