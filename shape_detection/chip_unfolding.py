from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    from geometry import PointArray, Line
    from chip_extraction import MainFeatures

from dataclasses import dataclass

import geometry
from shape_detection.chip_extraction import extract_main_features

import numpy as np
from numpy.polynomial import Polynomial
import cv2 as cv


@dataclass
class InnerChipFeatures:
    dorsal_pts: PointArray
    bisectors: Sequence[Line]


def compute_chip_convex_hull(main_ft: MainFeatures, chip_pts: PointArray) -> PointArray:
    """Compute the convex hull of the chip while constraining it to go through
    three anchor points.
    Return the convex hull points in the chip rotation order. The first point of
    the hull is the intersection between the tool and the base.
    """
    furthest_idx, _ = geometry.line_furthest_point(chip_pts, main_ft.tool_line)
    chip_furthest = chip_pts[furthest_idx, 0, :]

    anchor_0 = geometry.orthogonal_projection(*chip_furthest, main_ft.base_border)
    anchor_1 = main_ft.tool_base_intersection
    anchors = np.array([anchor_0, anchor_1], dtype=np.int32).reshape(-1, 1, 2)

    chip_hull_pts = cv.convexHull(np.vstack((chip_pts, anchors)), clockwise=main_ft.indirect_rotation)

    first_pt_idx = np.where(
        (chip_hull_pts[:, 0, 0] == anchor_1[0]) &
        (chip_hull_pts[:, 0, 1] == anchor_1[1])
    )[0][0]

    return np.roll(chip_hull_pts, -first_pt_idx, axis=0)[:-1]


def extract_chip_dorsal(main_ft: MainFeatures, chip_hull_pts: PointArray) -> PointArray:
    rho1, xn1, yn1 = main_ft.base_opp_border
    rho2, xn2, yn2 = main_ft.tool_opp_border

    margin = 15

    for i in range(len(chip_hull_pts)):
        x, y = chip_hull_pts[i, 0, 0], chip_hull_pts[i, 0, 1]
        if xn1*x + yn1*y - rho1 + margin > 0 or xn2*x + yn2*y - rho2 + margin > 0:
            return chip_hull_pts[:i]

    return chip_hull_pts


def compute_dorsal_bisectors(dorsal_pts: PointArray) -> Sequence[Line]:
    ...


def extract_inner_chip_features(binary_img: np.ndarray) -> tuple[MainFeatures, InnerChipFeatures]:
    main_ft = extract_main_features(binary_img)

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))

    chip_hull_pts = compute_chip_convex_hull(main_ft, chip_pts)
    chip_dorsal_pts = extract_chip_dorsal(main_ft, chip_hull_pts)
    bisectors = compute_dorsal_bisectors(chip_dorsal_pts)

    return main_ft, InnerChipFeatures(chip_dorsal_pts, bisectors)


def render_inner_chip_features(render: np.ndarray, main_ft: MainFeatures, inner_ft: InnerChipFeatures) -> None:
    red = (0, 0, 255)
    green = (0, 255, 0)

    geometry.draw_line(render, main_ft.base_line, color=red, thickness=3)
    geometry.draw_line(render, main_ft.tool_line, color=red, thickness=3)

    for pt in inner_ft.dorsal_pts.reshape(-1, 2):
        cv.circle(render, pt, 6, color=green, thickness=-1)


def extract_and_render(binary_img: np.ndarray) -> np.ndarray:
    main_ft, inner_ft = extract_inner_chip_features(binary_img)
    ft_repr = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
    render_inner_chip_features(ft_repr, main_ft, inner_ft)
    ft_repr[np.nonzero(binary_img)] = (255, 255, 255)
    return ft_repr


if __name__ == '__main__':
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt

    import image_loader
    import preprocessing.log_tresh_blobfilter_erode

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("unfolding", extract_and_render)

    input_dir_str = os.environ.get("INPUT_DIR")
    if input_dir_str is not None:
        input_dir = Path(os.environ["INPUT_DIR"])
    else:
        input_dir = Path("imgs", "vertical")

    output_dir = Path("results", "unfolding")
    loader = image_loader.ImageLoader(input_dir)

    processing.run(loader)
    processing.show_frame_comp(15, ("unfolding",))
    processing.show_video_comp(("unfolding",))
