import numpy as np

def positive_rho(rho, theta):
    """Return the polar parameters (rho, theta) converted such that rho >= 0."""
    if rho >= 0:
        return rho, theta
    else:
        return -rho, theta+np.pi


def above_line(points, rho, xn, yn, min_distance):
    """Keeps only the points above the line whose polar parameters are
    (rho, theta), with xn = cos(theta), yn = sin(theta).
    """
    x, y = points[:, 0, 0], points[:, 0, 1]
    mask = (xn*x + yn*y - rho - min_distance >= 0).flatten()
    return points[mask]


def under_line(points, rho, xn, yn, min_distance):
    """Keeps only the points under the line whose polar parameters are
    (rho, theta), with xn = cos(theta), yn = sin(theta).
    """
    x, y = points[:, 0, 0], points[:, 0, 1]
    mask = (xn*x + yn*y - rho + min_distance <= 0).flatten()
    return points[mask]


def intersect_line(rho0, xn0, yn0, rho1, xn1, yn1):
    """Return the intersection points of the two line whose polar parameters are
    (rho0, theta0) and (rho1, theta1), with
    xn0 = cos(theta0), yn0 = sin(theta0) and xn1 = cos(theta1), yn1 = sin(theta1).
    """
    denominator = yn0*xn1 - xn0*yn1
    return (
        int((rho1*yn0 - rho0*yn1) / denominator),
        int((rho0*xn1 - rho1*xn0) / denominator)
    )
