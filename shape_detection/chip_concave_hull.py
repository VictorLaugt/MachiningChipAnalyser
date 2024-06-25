from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geometry import PointArray
    from chip_extraction import MainFeatures

from dataclasses import dataclass

import sys

import geometry
from shape_detection.chip_extraction import extract_main_features
import concave_hull  # pip install concave_hull

import numpy as np
from numpy.polynomial import Polynomial
import cv2 as cv


@dataclass
class ChipFeatures:
    hull_pts: PointArray
    key_pts: PointArray
    polynomial: Polynomial
    contact_point: tuple[float, float]


def compute_chip_concave_hull(chip_pts: PointArray) -> PointArray:
    """Return the points of a concave hull of the chip."""
    chup_hull_idx = concave_hull.concave_hull_indexes(
        chip_pts.reshape(-1, 2),
        concavity=1.2,
        length_threshold=0.0
    )
    return chip_pts[chup_hull_idx].reshape(-1, 1, 2)


def extract_chip_curve_points(main_ft: MainFeatures, chip_hull_pts: PointArray) -> PointArray:
    """Return the points of the chip hull which belong to the chip curve."""
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
    rot_x, rot_y = geometry.rotate(key_pts[:, 0, 0], key_pts[:, 0, 1], -main_ft.tool_angle)

    if len(key_pts) < 2:
        print("Warning !: Cannot fit the chip curve", file=sys.stderr)
        polynomial = None
    elif len(key_pts) == 2:
        polynomial = Polynomial.fit(rot_x, rot_y, 1)
    else:
        polynomial = Polynomial.fit(rot_x, rot_y, 2)

    return polynomial


def chip_tool_contact_point(main_ft: MainFeatures, polynomial: Polynomial) -> tuple[float, float]:
    """Return the contact point between the tool and the chip curve."""
    # abscissa of the contact point, rotated in the polynomial basis
    xi, yi = main_ft.tool_base_intersection
    cos, sin = np.cos(main_ft.tool_angle), -np.sin(main_ft.tool_angle)
    rot_xc = xi*cos - yi*sin

    # ordinate of the contact point, rotated in the polynomial basis
    rot_yc = polynomial(rot_xc)

    # contact point, rotated back in the image basis
    return geometry.rotate(rot_xc, rot_yc, main_ft.tool_angle)


def extract_chip_features(binary_img: np.ndarray) -> tuple[MainFeatures, PointArray]:
    """Detect and return the chip features from the preprocessed binary image."""
    main_ft = extract_main_features(binary_img)

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))

    chip_hull_pts = compute_chip_concave_hull(chip_pts)
    # chip_curve_pts = extract_chip_curve_points(main_ft, chip_hull_pts)
    # key_pts = extract_key_points(main_ft, chip_curve_pts, np.pi/4)
    # polynomial = fit_polynomial(main_ft, key_pts)
    # contact = chip_tool_contact_point(main_ft, polynomial)

    return main_ft, chip_hull_pts


def render_chip_features(render: np.ndarray, main_ft: MainFeatures, chip_hull_pts: PointArray) -> None:
    """Draw a representation of features `main_ft` and `chip_ft` on image `render`."""
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    # dark_green = (0, 85, 0)
    # blue = (255, 0, 0)

    geometry.draw_line(render, main_ft.base_line, color=red, thickness=3)
    geometry.draw_line(render, main_ft.tool_line, color=red, thickness=3)

    for a, b in zip(chip_hull_pts[:-1], chip_hull_pts[1:]):
        cv.line(render, tuple(a[0]), tuple(b[0]), color=yellow, thickness=1)
        # cv.circle(render, tuple(a[0]), 6, color=green, thickness=-1)


class ChipFeatureCollector:
    def __init__(self, scale: float=1.0):
        pass

    def extract_and_render(self, binary_img: np.ndarray, background: np.ndarray=None) -> np.ndarray:
        main_ft, chip_hull_pts = extract_chip_features(binary_img)
        if background is None:
            ft_repr = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
            ft_repr[np.nonzero(binary_img)] = (255, 255, 255)
            render_chip_features(ft_repr, main_ft, chip_hull_pts)
        else:
            ft_repr = background.copy()
            render_chip_features(ft_repr, main_ft, chip_hull_pts)
        return ft_repr


if __name__ == '__main__':
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt

    import image_loader
    import preprocessing.log_tresh_blobfilter_erode

    collector = ChipFeatureCollector()

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    # processing.add("chipcurve", collector.extract_and_render, ("morph", "input"))
    processing.add("chipcurve", collector.extract_and_render)

    input_dir_str = os.environ.get("INPUT_DIR")
    if input_dir_str is not None:
        input_dir = Path(os.environ["INPUT_DIR"])
    else:
        input_dir = Path("imgs", "vertical")

    output_dir = Path("results", "chipcurve")
    loader = image_loader.ImageLoader(input_dir)

    processing.run(loader)
    processing.show_frame_comp(15, ("chipcurve", "input"))
    processing.show_video_comp(("chipcurve", "input"))
