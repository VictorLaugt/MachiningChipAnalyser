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
    h, w = 500, 900
    img = np.zeros((h, w, 3), dtype=np.uint8)

    edge_points = np.array([[653, 183], [591, 147], [341, 149], [8, 482]]).reshape(-1, 1, 2)

    edge_lines = []
    for i in range(len(edge_points)-1):
        a, b = edge_points[i, 0], edge_points[i+1, 0]
        edge_lines.append(geometry.line_from_two_points(a, b))

    n_points = 1000
    points = np.column_stack((
        np.random.randint(0, w + 1, n_points),
        np.random.randint(0, h + 1, n_points)
    )).reshape(-1, 1, 2)


    labels = classify_by_nearest_edge(points, edge_lines)
    groups = [points[np.where(labels == lbl)] for lbl in range(len(edge_lines))]

    group_colors = ((127, 0, 0), (0, 127, 0), (0, 0, 127))
    edge_colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
    for color, group in zip(group_colors, groups):
        for pt in group:
            cv.circle(img, (pt[0, 0], pt[0, 1]), 3, color, -1)
    for color, edge in zip(edge_colors, edge_lines):
        geometry.draw_line(img, edge, color, 5)

    cv.imshow('img', img)
    while cv.waitKey(0) != 113:
        pass
    cv.destroyAllWindows()
