from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Iterable, Optional
    from type_hints import ColorImage, GrayImage, IntPtArray, IntPt, Line

import cv2 as cv
import numpy as np


NAN_LINE: Line = (np.nan, np.nan, np.nan)


def draw_line(img: GrayImage|ColorImage, line: Line, color: int|tuple[int, int], thickness: int) -> None:
    """Draw a line on an image, with a given color and a given thickness.

    Possible signatures
    -------------------
    (GrayImage, Line, int, int) -> None
    (ColorImage, Line, tuple[int, int, int], int) -> None
    """
    rho, xn, yn = line
    x0, y0 = rho * xn, rho * yn
    x1, y1 = int(x0 - 2000 * yn), int(y0 + 2000 * xn)
    x2, y2 = int(x0 + 2000 * yn), int(y0 - 2000 * xn)
    cv.line(img, (x1, y1), (x2, y2), color, thickness)


def rotate(x: Any, y: Any, angle: float) -> tuple[Any, Any]:
    """Rotate a point or a point array by a given angle in radians.

    Possible signatures
    -------------------
    (int|float, int|float) -> tuple[float, float]
    (IntArray|FloatArray, IntArray|FloatArray, float) -> tuple[FloatArray, FloatArray]
    """
    cos, sin = np.cos(angle), np.sin(angle)
    return (x*cos - y*sin, x*sin + y*cos)


def orthogonal_projection(x: Any, y: Any, line: Line) -> tuple[Any, Any]:
    """Compute the orthogonal projection of a point or a point array on a line.

    Possible signatures
    -------------------
    (int|float, int|float, Line) -> tuple[float, float]
    (IntArray|FloatArray, IntArray|FloatArray, Line) -> tuple[FloatArray, FloatArray]
    """
    rho, xn, yn = line
    signed_dist = xn*x + yn*y - rho
    return (x - signed_dist*xn, y - signed_dist*yn)


def standard_polar_param(rho: float, theta: float) -> tuple[float, float]:
    """Standardize the polar parameters (rho, theta)
    Input:  -inf < rho < +inf and 0 <= theta < pi
    Output:   0 <= rho < +inf and 0 <= theta < 2*pi
    """
    if rho >= 0:
        return (rho, theta)
    else:
        return (-rho, theta+np.pi)


def neg_line(line: Line) -> Line:
    """Return the opposite line, i.e the line with flipped above and under sides."""
    rho, xn, yn = line
    return (-rho, -xn, -yn)


def above_lines(points: IntPtArray, lines: Iterable[Line], margins: Iterable[int]) -> IntPtArray:
    """Keep only the points above the given lines with given margins."""
    x, y = points[:, 0], points[:, 1]
    mask = np.ones(len(points), dtype=bool)
    for (rho, xn, yn), min_distance in zip(lines, margins):
        mask &= (xn*x + yn*y - rho - min_distance >= 0).flatten()
    return points[mask]


def under_lines(points: IntPtArray, lines: Iterable[Line], margins: Iterable[int]) -> IntPtArray:
    """Keep only the points under the given lines with given margins."""
    x, y = points[:, 0], points[:, 1]
    mask = np.ones(len(points), dtype=bool)
    for (rho, xn, yn), min_distance in zip(lines, margins):
        mask &= (xn*x + yn*y - rho + min_distance <= 0).flatten()
    return points[mask]


def line_furthest_point(points: IntPtArray, line: Line) -> tuple[int, float]:
    """Return the index of the furthest point to the line and its distance to
    the line.
    """
    rho, xn, yn = line
    x, y = points[:, 0], points[:, 1]
    distances = np.abs(xn*x + yn*y - rho)
    i = np.argmax(distances)
    return i, distances[i]


def intersect_line(line0: Line, line1: Line) -> IntPt:
    """Compute the intersection point of two lines."""
    (rho0, xn0, yn0), (rho1, xn1, yn1) = line0, line1
    det = yn0*xn1 - xn0*yn1
    return (
        int((rho1*yn0 - rho0*yn1) / det),
        int((rho0*xn1 - rho1*xn0) / det)
    )

def intersect_line_safe(line0: Line, line1: Line) -> Optional[IntPt]:
    """Compute the intersection point of two lines and return None if it doesn't exist."""
    (rho0, xn0, yn0), (rho1, xn1, yn1) = line0, line1
    det = yn0*xn1 - xn0*yn1
    if det != 0:
        return (
            int((rho1*yn0 - rho0*yn1) / det),
            int((rho0*xn1 - rho1*xn0) / det)
        )
