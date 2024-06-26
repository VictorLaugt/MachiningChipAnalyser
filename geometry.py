from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import TypeVar, Iterable
    T = TypeVar('T', int, float, np.ndarray)
    PointArray = TypeVar('PointArray', bound=np.ndarray)  # ~ (n, 1, 2) dtype=int32
    PolarLine = tuple[float, float, float]  # ~ (rho, xn, yn), xn = cos(theta), yn = sin(theta)

import cv2 as cv
import numpy as np


def standard_polar_param(rho: float, theta: float) -> tuple[float, float]:
    """Standardize the polar parameters (rho, theta)
    Input:  -inf < rho < +inf and 0 <= theta < pi
    Output:   0 <= rho < +inf and 0 <= theta < 2*pi
    """
    if rho >= 0:
        return (rho, theta)
    else:
        return (-rho, theta+np.pi)


def rotate(x: T, y: T, angle: T) -> tuple[T, T]:
    """Return the opposite line, i.e the line with flipped above and under
    sides.
    """
    cos, sin = np.cos(angle), np.sin(angle)
    return (x*cos - y*sin, x*sin + y*cos)


def neg_line(line: PolarLine) -> PolarLine:
    rho, xn, yn = line
    return (-rho, -xn, -yn)


def above_lines(points: PointArray, lines: Iterable[PolarLine], margins: Iterable[int]) -> PointArray:
    """Keeps only the points above the lines with a margin."""
    x, y = points[:, 0, 0], points[:, 0, 1]
    mask = np.ones(len(points), dtype=bool)
    for (rho, xn, yn), min_distance in zip(lines, margins):
        mask &= (xn*x + yn*y - rho - min_distance >= 0).flatten()
    return points[mask]


def under_lines(points: PointArray, lines: Iterable[PolarLine], margins: Iterable[int]) -> PointArray:
    """Keeps only the points under the lines with a margin."""
    x, y = points[:, 0, 0], points[:, 0, 1]
    mask = np.ones(len(points), dtype=bool)
    for (rho, xn, yn), min_distance in zip(lines, margins):
        mask &= (xn*x + yn*y - rho + min_distance <= 0).flatten()
    return points[mask]


def line_nearest_point(points: PointArray, line: PolarLine) -> tuple[int, float]:
    """Return the index of the point nearest to the line and its distance to the
    line.
    """
    rho, xn, yn = line
    x, y = points[:, 0, 0], points[:, 0, 1]
    distances = np.abs(xn*x + yn*y - rho)
    i = np.argmin(distances)
    return i, distances[i]


def line_furthest_point(points: PointArray, line: PolarLine) -> tuple[int, float]:
    """Return the index of the point furthest to the line and its distance to
    the line.
    """
    rho, xn, yn = line
    x, y = points[:, 0, 0], points[:, 0, 1]
    distances = np.abs(xn*x + yn*y - rho)
    i = np.argmax(distances)
    return i, distances[i]


def orthogonal_projection(x: T, y: T, line: PolarLine) -> tuple[T, T]:
    """Compute the orthogonal projection of a point on a line."""
    rho, xn, yn = line
    dist = xn*x + yn*y - rho
    return (x - dist*xn, y - dist*yn)


def intersect_line(line0: PolarLine, line1: PolarLine) -> tuple[int, int]:
    """Compute the intersection points of two lines."""
    (rho0, xn0, yn0), (rho1, xn1, yn1) = line0, line1
    denominator = yn0*xn1 - xn0*yn1
    return (
        int((rho1*yn0 - rho0*yn1) / denominator),
        int((rho0*xn1 - rho1*xn0) / denominator)
    )


def draw_line(img: np.ndarray, line: PolarLine, color: int, thickness: int) -> None:
    """Draw on img a line."""
    rho, xn, yn = line
    x0, y0 = rho * xn, rho * yn
    x1, y1 = int(x0 - 2000 * yn), int(y0 + 2000 * xn)
    x2, y2 = int(x0 + 2000 * yn), int(y0 - 2000 * xn)
    cv.line(img, (x1, y1), (x2, y2), color, thickness)


def parallel(line: PolarLine, x: float, y: float) -> PolarLine:
    """Return the line parallel to the input line and passing through the point
    (x, y).
    """
    _rho, xn, yn = line
    return (x*xn + y*yn, xn, yn)
