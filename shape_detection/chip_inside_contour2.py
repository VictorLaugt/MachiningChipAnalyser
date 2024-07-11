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
import skimage as ski



@dataclass
class ChipInsideFeatures:
    thickness: Sequence[float]
    inside_contour_pts: Sequence[Point]


def direct_rotated_90(p: Point) -> Point:
    x, y = p
    return (-y, x)

def indirect_rotated_90(p: Point) -> Point:
    x, y = p
    return (y, -x)

def normalized(vector: Point) -> Point:
    x, y = vector
    norm = np.linalg.norm(vector)
    return (x / norm, y / norm)


def rasterized_line(p0: Point, p1: Point, img_height: int, img_width: int) -> tuple[np.ndarray[int], np.ndarray[int]]:
    line_x, line_y = ski.draw.line(*p0, *p1)
    inside_mask = (0 <= line_x) & (line_x < img_width) & (0 <= line_y) & (line_y < img_height)
    return line_x[inside_mask], line_y[inside_mask]


def find_inside_contour(
    binary_img: np.ndarray,
    chip_curve_pts: PointArray,
    thickness_majorant: int
        ) -> ChipInsideFeatures:
    h, w = binary_img.shape
    thickness = []
    inside_contour_pts = []

    for i in range(len(chip_curve_pts)-1):
        a, b = chip_curve_pts[i, 0, :], chip_curve_pts[i+1, 0, :]
        n = np.asarray(normalized(indirect_rotated_90(b - a)))
        c, d = (a + thickness_majorant*n).astype(np.int32), (b + thickness_majorant*n).astype(np.int32)

        for p0, p1 in zip(zip(*rasterized_line(a, b, h, w)), zip(*rasterized_line(c, d, h, w))):
            ray_x, ray_y = rasterized_line(p0, p1, h, w)

            selected_idx = np.nonzero(binary_img[ray_y, ray_x])[0]
            if len(selected_idx) > 0:
                selected_x, selected_y = ray_x[selected_idx], ray_y[selected_idx]

                distances = np.linalg.norm((selected_x - p0[0], selected_y - p0[1]), axis=0)
                furthest_idx = np.argmax(distances)

                thickness.append(distances[furthest_idx])
                inside_contour_pts.append((selected_x[furthest_idx], selected_y[furthest_idx]))

    return ChipInsideFeatures(thickness, inside_contour_pts)


def extract_chip_inside_contour(binary_img: np.ndarray) -> tuple[MainFeatures, ChipInsideFeatures]:
    main_ft = extract_main_features(binary_img)

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))

    chip_hull_pts = compute_chip_convex_hull(main_ft, chip_pts)
    chip_curve_pts = extract_chip_curve_points(main_ft, chip_hull_pts)

    inside_ft = find_inside_contour(binary_img, chip_curve_pts, thickness_majorant=125)

    return main_ft, inside_ft


def render_inside_features(render: np.ndarray, main_ft: MainFeatures, inside_ft: ChipInsideFeatures) -> None:
    """Draw a representation of features `main_ft` and `inside_ft` on image `render`."""
    for x, y in inside_ft.inside_contour_pts:
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
