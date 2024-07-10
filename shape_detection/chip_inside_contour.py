from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence
    from chip_extraction import MainFeatures
    from geometry import Line, Point, PointArray

from dataclasses import dataclass

import geometry
from shape_detection.chip_extraction import extract_main_features

import numpy as np
import cv2 as cv


@dataclass
class ChipInsideFeatures:
    thickness: Sequence[float]
    inside_pts: Sequence[Point]


def compute_chip_convex_hull(main_ft: MainFeatures, chip_pts: PointArray) -> PointArray:
    """Compute the convex hull of the chip while constraining it to go through
    three anchor points.
    Return the convex hull points in the chip rotation order. The first point of
    the hull is the intersection between the tool and the base.
    """
    highest_idx, _ = geometry.line_furthest_point(chip_pts, main_ft.base_line)
    chip_highest = chip_pts[highest_idx, 0, :]

    anchor_0 = geometry.orthogonal_projection(*chip_highest, main_ft.tool_opp_border)
    anchor_1 = geometry.orthogonal_projection(*anchor_0, main_ft.base_border)
    anchor_2 = main_ft.tool_base_intersection
    anchors = np.array([anchor_0, anchor_1, anchor_2], dtype=np.int32).reshape(-1, 1, 2)

    chip_hull_pts = cv.convexHull(np.vstack((chip_pts, anchors)), clockwise=main_ft.indirect_rotation)

    first_pt_idx = np.where(
        (chip_hull_pts[:, 0, 0] == anchor_2[0]) &
        (chip_hull_pts[:, 0, 1] == anchor_2[1])
    )[0][0]

    return np.roll(chip_hull_pts, -first_pt_idx, axis=0)


def extract_chip_curve_points(main_ft: MainFeatures, chip_hull_pts: PointArray) -> PointArray:
    """Return the points of the chip hull which belong to the chip curve."""
    # _, base_distance = geometry.line_nearest_point(chip_hull_pts, main_ft.base_line)
    # _, tool_distance = geometry.line_nearest_point(chip_hull_pts, main_ft.tool_line)

    # return geometry.under_lines(
    #     chip_hull_pts,
    #     (main_ft.base_line, main_ft.tool_line, main_ft.base_opp_border, main_ft.tool_opp_border),
    #     (base_distance+20, tool_distance+5, 15, 15)
    # )

    return geometry.under_lines(
        chip_hull_pts[1:],
        (main_ft.base_line, main_ft.base_opp_border, main_ft.tool_opp_border),
        (0, 15, 15)
    )


def create_edge_lines(chip_curve_pts: PointArray) -> tuple[np.ndarray, np.ndarray]:
    """Create the edge lines of the chip curve."""
    edge_lines = []
    for i in range(len(chip_curve_pts)-1):
        a, b = chip_curve_pts[i, 0, :], chip_curve_pts[i+1, 0, :]
        edge_lines.append(geometry.line_from_two_points(a, b))
    return edge_lines


def compute_distance_edge_points(chip_pts: PointArray, edge_lines: Sequence[Line]) -> np.ndarray[float]:
    dist_edge_pt = np.empty((len(edge_lines), len(chip_pts)), dtype=np.float32)
    for i, edge in enumerate(edge_lines):
        dist_edge_pt[i, :] = geometry.line_points_distance(chip_pts, edge)
    return dist_edge_pt


def find_inside_contour(
            chip_pts: PointArray,
            edge_lines: Sequence[Line],
            nearest_edge_idx: np.ndarray[int]
        ) -> ChipInsideFeatures:
    opposite = {}
    thickness = {}

    for j in range(len(chip_pts)):
        p = chip_pts[j, 0, :]
        nearest_edge = edge_lines[nearest_edge_idx[j]]
        dist, (xe, ye) = geometry.dist_orthogonal_projection(p, nearest_edge)
        e = (int(xe), int(ye))

        max_dist = thickness.get(e, -1.)
        if dist > max_dist:
            thickness[e] = dist
            opposite[e] = p

    return ChipInsideFeatures(list(thickness.values()), list(opposite.values()))


def extract_chip_inside_contour(binary_img: np.ndarray) -> tuple[MainFeatures, ChipInsideFeatures]:
    main_ft = extract_main_features(binary_img)

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))

    chip_hull_pts = compute_chip_convex_hull(main_ft, chip_pts)
    chip_curve_pts = extract_chip_curve_points(main_ft, chip_hull_pts)

    edge_lines = create_edge_lines(chip_curve_pts)
    dist_edge_pt = compute_distance_edge_points(chip_pts, edge_lines)
    nearest_edge_idx = np.argmin(dist_edge_pt, axis=0)

    return main_ft, find_inside_contour(chip_pts, edge_lines, nearest_edge_idx)


def render_inside_features(render: np.ndarray, main_ft: MainFeatures, inside_ft: ChipInsideFeatures) -> None:
    """Draw a representation of features `main_ft` and `inside_ft` on image `render`."""
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    # green = (0, 255, 0)
    # dark_green = (0, 85, 0)
    # blue = (255, 0, 0)

    # geometry.draw_line(render, main_ft.base_line, color=red, thickness=3)
    # geometry.draw_line(render, main_ft.tool_line, color=red, thickness=3)

    for x, y in inside_ft.inside_pts:
        render[y, x] = yellow


def extract_and_render(binary_img: np.ndarray, background: np.ndarray|None=None) -> np.ndarray:
    main_ft, inside_ft = extract_chip_inside_contour(binary_img)
    if background is None:
        ft_repr = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
        render_inside_features(ft_repr, main_ft, inside_ft)
        # ft_repr[np.nonzero(binary_img)] = (255, 255, 255)
    else:
        ft_repr = background.copy()
        render_inside_features(ft_repr, main_ft, inside_ft)
    return ft_repr


if __name__ == '__main__':
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt

    import image_loader
    import preprocessing.log_tresh_blobfilter_erode

    # ---- environment variables
    input_dir_str = os.environ.get("INPUT_DIR")
    output_dir_str = os.environ.get("OUTPUT_DIR")

    if input_dir_str is not None:
        input_dir = Path(input_dir_str)
    else:
        input_dir = Path("imgs", "vertical")

    if output_dir_str is not None:
        output_dir = Path(output_dir_str)
    else:
        output_dir = Path("results", "chipvurve")


    # ---- processing
    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()

    processing.add("chipinside", extract_and_render)

    loader = image_loader.ImageLoader(input_dir)
    processing.run(loader)


    # ---- visualization
    processing.show_frame_comp(min(15, len(loader)-1), ("chipinside", "input"))
    processing.show_video_comp(("chipinside", "input"))
