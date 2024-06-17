from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geometry import Line, PointArray
    from chip_extraction import MainFeatures

from dataclasses import dataclass

import sys

import geometry
from shape_detection.chip_extraction import extract_main_features

import numpy as np
from numpy.polynomial import Polynomial
import cv2 as cv


@dataclass
class ChipFeatures:
    polynomial: Polynomial
    hull_pts: PointArray
    key_pts: PointArray


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


def extract_key_points(main_ft: MainFeatures, curve_points: PointArray, tool_chip_max_angle: float) -> PointArray:
    """Return key points from the chip curve which can then be use to fit a polynomial."""
    if main_ft.tool_angle > np.pi:
        tool_angle = main_ft.tool_angle - 2*np.pi
    else:
        tool_angle = main_ft.tool_angle

    if main_ft.indirect_rotation:
        curve_surface_vectors = curve_points[:-1, 0, :] - curve_points[1:, 0, :]
    else:
        curve_surface_vectors = curve_points[1:, 0, :] - curve_points[:-1, 0, :]

    curve_segment_lengths = np.linalg.norm(curve_surface_vectors, axis=-1)
    curve_vector_angles = np.arccos(curve_surface_vectors[:, 0] / curve_segment_lengths)

    mask = np.ones(len(curve_points), dtype=bool)     # keep the first point
    # mask = np.zeros(len(curve_points), dtype=bool)  # exclude the first point
    mask[1:] = np.abs(np.pi/2 + tool_angle - curve_vector_angles) < tool_chip_max_angle

    return curve_points[mask]


def fit_polynomial(main_ft: MainFeatures, key_pts: PointArray) -> Polynomial:
    """Return a polynomial of degree 2 which fits the key points."""
    x, y = geometry.rotate(key_pts[:, 0, 0], key_pts[:, 0, 1], -main_ft.tool_angle)
    if len(key_pts) < 2:
        print("Warning !: Chip curve not found", file=sys.stderr)
        polynomial = None
    elif len(key_pts) == 2:
        polynomial = Polynomial.fit(x, y, 1)
    else:
        polynomial = Polynomial.fit(x, y, 2)

    return polynomial


def extract_chip_features(binary_img: np.ndarray) -> tuple[MainFeatures, ChipFeatures]:
    """Detect and return the chip features from the preprocessed binary image."""
    main_ft = extract_main_features(binary_img)

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))

    chip_hull_pts = compute_chip_convex_hull(main_ft, chip_pts)
    chip_curve_pts = extract_chip_curve_points(main_ft, chip_hull_pts)
    key_pts = extract_key_points(main_ft, chip_curve_pts, np.pi/4)
    polynomial = fit_polynomial(main_ft, key_pts)

    return main_ft, ChipFeatures(polynomial, chip_hull_pts, key_pts)


def render_chip_features(binary_img: np.ndarray, render=None) -> np.ndarray:
    """Return an image which shows on `render` the detected features from the
    preprocessed binary image `binary_img`.
    """
    h, w = binary_img.shape

    if render is None:
        render = np.zeros_like(binary_img)
    else:
        render = render.copy()

    main_ft, chip_ft = extract_chip_features(binary_img)

    if chip_ft.polynomial is not None:
        x = np.arange(0, w, 1, dtype=np.int32)
        y = chip_ft.polynomial(x)
        x, y = geometry.rotate(x, y, main_ft.tool_angle)
        x, y = x.astype(np.int32), y.astype(np.int32)
        inside_mask = (2 <= x) & (x < w-2) & (2 <= y) & (y < h-2)
        x, y = x[inside_mask], y[inside_mask]
        render[y, x] = render[y, x+1] = render[y, x-1] = render[y+1, x] = render[y-1, x] = 127

    for pt in chip_ft.hull_pts.reshape(-1, 2):
        cv.circle(render, pt, 4, color=127//2, thickness=-1)
    for kpt in chip_ft.key_pts.reshape(-1, 2):
        cv.circle(render, kpt, 4, color=127, thickness=-1)

    geometry.draw_line(render, main_ft.base_line, color=127, thickness=2)
    geometry.draw_line(render, main_ft.tool_line, color=127, thickness=2)

    return render


if __name__ == '__main__':
    import os
    from pathlib import Path

    import image_loader
    import preprocessing.log_tresh_blobfilter_erode

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("chipcurve", render_chip_features, ("morph", "morph"))

    input_dir_str = os.environ.get("INPUT_DIR")
    if input_dir_str is not None:
        input_dir = Path(os.environ["INPUT_DIR"])
    else:
        input_dir = Path("imgs", "vertical")

    output_dir = Path("results", "chipcurve")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    processing.run(loader, output_dir)
    processing.compare_frames(15, ("chipcurve", "input"))
    processing.compare_videos(("chipcurve", "input"))


# if __name__ == '__main__':
#     img = cv.cvtColor(cv.imread('preprocessed.png'), cv.COLOR_RGB2GRAY)
#     extracted = render_chip_curve(img, img, render=img)

#     cv.imshow('extracted', extracted)
#     while cv.waitKey(30) != 113:
#         pass
#     cv.destroyAllWindows()
