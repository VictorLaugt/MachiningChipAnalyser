from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    from geometry import Line, Point, PointArray
    from shape_detection.chip_extraction import MainFeatures

import geometry
from shape_detection.constrained_hull_polynomial import (
    compute_chip_convex_hull,
    extract_chip_curve_points
)
from shape_detection.chip_extraction import extract_main_features

import numpy as np
import cv2 as cv

from dataclasses import dataclass


@dataclass
class ChipInsideFeatures:
    thickness: Sequence[float]
    inside_pts: Sequence[Point]


def create_edge_lines(chip_curve_pts: PointArray) -> Sequence[Line]:
    edge_lines = []
    for i in range(len(chip_curve_pts)-1):
        a, b = chip_curve_pts[i, 0, :], chip_curve_pts[i+1, 0, :]
        edge_lines.append(geometry.line_from_two_points(a, b))
    return edge_lines


def compute_distance_edge_points(chip_pts: PointArray, edge_lines: Sequence[Line]) -> np.ndarray[float]:
    # dist_edge_pt[i, j] == distance from edge_lines[i] to chip_pts[j]
    dist_edge_pt = np.empty((len(edge_lines), len(chip_pts)), dtype=np.float32)
    for i, edge in enumerate(edge_lines):
        dist_edge_pt[i, :] = geometry.line_points_distance(chip_pts, edge)
    return dist_edge_pt


def find_inside_contour(
            chip_pts: PointArray,
            edge_lines: Sequence[Line],
            nearest_edge_idx: np.ndarray[int],
            max_thickness: float
        ) -> ChipInsideFeatures:
    opposite = {}
    thickness = {}

    for j in range(len(chip_pts)):
        p = chip_pts[j, 0, :]
        nearest_edge = edge_lines[nearest_edge_idx[j]]
        dist, (xe, ye) = geometry.dist_orthogonal_projection(p, nearest_edge)
        e = (int(xe), int(ye))

        max_dist = thickness.get(e, -1.)
        if max_dist < dist < max_thickness:
            thickness[e] = dist
            opposite[e] = p

    return ChipInsideFeatures(list(thickness.values()), list(opposite.values()))


def render_chip_inside(binary_img: np.ndarray) -> np.ndarray:
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    dark_green = (0, 85, 0)
    blue = (255, 0, 0)
    ft_repr = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
    # ft_repr[np.nonzero(binary_img)] = (255, 255, 255)

    main_ft = extract_main_features(binary_img)

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))

    chip_hull_pts = compute_chip_convex_hull(main_ft, chip_pts)
    chip_curve_pts = extract_chip_curve_points(main_ft, chip_hull_pts)

    edge_lines = create_edge_lines(chip_curve_pts)
    dist_edge_pt = compute_distance_edge_points(chip_pts, edge_lines)
    nearest_edge_idx = np.argmin(dist_edge_pt, axis=0)
    inside_ft = find_inside_contour(chip_pts, edge_lines, nearest_edge_idx, max_thickness=125.)

    for pt in chip_curve_pts.reshape(-1, 2):
        cv.circle(ft_repr, pt, 3, color=green, thickness=-1)
    for edge in edge_lines:
        geometry.draw_line(ft_repr, edge, yellow, 1)
    for x, y in inside_ft.inside_pts:
        ft_repr[y, x] = red

    return ft_repr


if __name__ == '__main__':
    from pathlib import Path

    img = cv.imread(str(Path('experiments', 'preprocessed_machining_image.png')))
    binary_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    inside_render = render_chip_inside(binary_img)

    cv.imshow('inside', inside_render)
    while cv.waitKey(0) != 113:
        pass
    cv.destroyAllWindows()
