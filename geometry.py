import cv2 as cv
import numpy as np


# def standard_polar_param(rho, theta):
#     """Standardize the polar parameters (rho, theta)
#     Input:  -inf < rho < +inf and  0  <= theta < pi
#     Output:   0 <= rho < +inf and -pi <= theta < pi
#     """
#     if rho >= 0:
#         return rho, theta
#     else:
#         return -rho, theta - np.pi


def standard_polar_param(rho, theta):
    """Standardize the polar parameters (rho, theta)
    Input:  -inf < rho < +inf and 0 <= theta < pi
    Output:   0 <= rho < +inf and 0 <= theta < 2*pi
    """
    if rho >= 0:
        return rho, theta
    else:
        return -rho, theta+np.pi


def rotate(x, y, angle):
    cos, sin = np.cos(angle), np.sin(angle)
    return x*cos - y*sin, x*sin + y*cos


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


def line_nearest_point(points, line):
    """Return the index of the point nearest to the line and its distance to the
    line. The line is described as (rho, xn, yn) where xn = cos(theta),
    yn = sin(theta), and (rho, theta) are the polar parameters.
    """
    rho, xn, yn = line
    x, y = points[:, 0, 0], points[:, 0, 1]
    distances = np.abs(xn*x + yn*y - rho)
    i = np.argmin(distances)
    return i, distances[i]

def line_furthest_point(points, line):
    """Return the index of the point furthest to the line and its distance to the
    line. The line is described as (rho, xn, yn) where xn = cos(theta),
    yn = sin(theta), and (rho, theta) are the polar parameters.
    """
    rho, xn, yn = line
    x, y = points[:, 0, 0], points[:, 0, 1]
    distances = np.abs(xn*x + yn*y - rho)
    i = np.argmax(distances)
    return i, distances[i]

def orthogonal_projection(x, y, line):
    """Compute the orthogonal projection of a point on a line.
    The line is described as (rho, xn, yn) where xn = cos(theta), yn = sin(theta),
    and (rho, theta) are the polar parameters.
    """
    rho, xn, yn = line
    dist = xn*x + yn*y - rho
    return (x - dist*xn, y - dist*yn)

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
