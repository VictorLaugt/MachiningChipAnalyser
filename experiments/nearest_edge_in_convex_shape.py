from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    from geometry import PointArray, Line

import numpy as np
import cv2 as cv

import geometry


def line_points_distance(points: PointArray, line: Line) -> np.ndarray[float]:
    rho, xn, yn = line
    x, y = points[:, 0, 0], points[:, 0, 1]
    return np.abs(xn*x + yn*y - rho)


def classify_by_nearest_edge(points: PointArray, edge_lines: Sequence[Line]) -> Sequence[np.ndarray[int]]:
    x, y = points[:, 0, 0], points[:, 0, 1]

    dist_to_edges = []
    for line in edge_lines:
        rho, xn, yn = line
        dist_to_edges.append(np.abs(xn*x + yn*y - rho))

    return np.argmin(dist_to_edges, axis=0)


if __name__ == '__main__':
    points = ...
    edge_lines = ...
    groups = points[classify_by_nearest_edge(points, edge_lines)]

    for edge, pts in zip(edge_lines, groups):
        ...




if __name__ == '__main__':
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    colors = np.array(((127, 0, 0), (0, 127, 0), (0, 0, 127)))

    h, w = 500, 900
    img = np.zeros((h, w, 3), dtype=np.uint8)

    factor = 50.
    rho_0, theta_0 = factor * 4 * 3**0.5, np.pi/4
    rho_1, theta_1 = factor * 3, np.pi/2
    rho_2, theta_2 = factor * -11.25**0.5, -np.pi/3 + np.pi

    line_0 = (rho_0, np.cos(theta_0), np.sin(theta_0))
    line_1 = (rho_1, np.cos(theta_1), np.sin(theta_1))
    line_2 = (rho_2, np.cos(theta_2), np.sin(theta_2))

    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    pts = np.column_stack((x.flatten(), y.flatten())).reshape(-1, 1, 2)
    pts = geometry.above_lines(pts, (line_0, line_1, line_2), (0, 0, 0))

    dist_0 = line_points_distance(pts, line_0)
    dist_1 = line_points_distance(pts, line_1)
    dist_2 = line_points_distance(pts, line_2)

    nearest_edge_indices = np.argmin((dist_0, dist_1, dist_2), axis=0)
    img[pts[:, 0, 1], pts[:, 0, 0]] = colors[nearest_edge_indices]

    geometry.draw_line(img, line_0, blue, 10)
    geometry.draw_line(img, line_1, green, 10)
    geometry.draw_line(img, line_2, red, 10)

    cv.imshow('img', img)
    while cv.waitKey(0) != 113:
        pass
    cv.destroyAllWindows()
