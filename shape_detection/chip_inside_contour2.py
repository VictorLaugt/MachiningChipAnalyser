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

from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
import cv2 as cv



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
            thickness_majorant: float
        ) -> ChipInsideFeatures:
    thickness = []
    inside_pts = []
    ...
    return ChipInsideFeatures(thickness, inside_pts)


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
    inside_ft = find_inside_contour(chip_pts, edge_lines, nearest_edge_idx, thickness_majorant=125.)

    return main_ft, inside_ft


def render_inside_features(render: np.ndarray, main_ft: MainFeatures, inside_ft: ChipInsideFeatures) -> None:
    """Draw a representation of features `main_ft` and `inside_ft` on image `render`."""
    for x, y in inside_ft.inside_pts:
        render[y, x] = (0, 0, 255)  # red


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
    processing.show_frame_comp(min(15, len(loader)-1), ("chipinside", "morph"))
    processing.show_video_comp(("chipinside", "morph"))
