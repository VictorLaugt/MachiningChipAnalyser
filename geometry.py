import cv2 as cv
import numpy as np


def positive_rho(rho, theta):
    """Return the polar parameters (rho, theta) converted such that rho >= 0."""
    if rho >= 0:
        return rho, theta
    else:
        return -rho, theta+np.pi


def above_lines(points, lines, margins):
    """Keeps only the points above the lines with a margin.
    The lines are described as (rho, xn, yn) where xn = cos(theta), yn = sin(theta),
    and (rho, theta) are the polar parameters.
    """
    x, y = points[:, 0, 0], points[:, 0, 1]
    mask = np.ones(len(points), dtype=bool)
    for (rho, xn, yn), min_distance in zip(lines, margins):
        mask &= (xn*x + yn*y - rho - min_distance >= 0).flatten()
    return points[mask]


def under_lines(points, lines, margins):
    """Keeps only the points under the lines with a margin.
    The lines are described as (rho, xn, yn) where xn = cos(theta), yn = sin(theta),
    and (rho, theta) are the polar parameters.
    """
    x, y = points[:, 0, 0], points[:, 0, 1]
    mask = np.ones(len(points), dtype=bool)
    for (rho, xn, yn), min_distance in zip(lines, margins):
        mask &= (xn*x + yn*y - rho + min_distance <= 0).flatten()
    return points[mask]


def intersect_line(line0, line1):
    """Compute the intersection points of two lines.
    The lines are described as (rho, xn, yn) where xn = cos(theta), yn = sin(theta),
    and (rho, theta) are the polar parameters.
    """
    (rho0, xn0, yn0), (rho1, xn1, yn1) = line0, line1
    denominator = yn0*xn1 - xn0*yn1
    return (
        int((rho1*yn0 - rho0*yn1) / denominator),
        int((rho0*xn1 - rho1*xn0) / denominator)
    )

def draw_line(img, line, color, thickness):
    """Draw on img a line.
    The lines are described as (rho, xn, yn) where xn = cos(theta), yn = sin(theta),
    and (rho, theta) are the polar parameters.
    """
    rho, xn, yn = line
    x0, y0 = rho * xn, rho * yn
    x1, y1 = int(x0 - 2000 * yn), int(y0 + 2000 * xn)
    x2, y2 = int(x0 + 2000 * yn), int(y0 - 2000 * xn)
    cv.line(img, (x1, y1), (x2, y2), color, thickness)
