from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
    from type_hints import GrayImage, OpenCVIntArray
    from chip_extract import MainFeatures
    from contact_measurement import ContactFeatures
    from thickness_measurement import InsideFeatures

import numpy as np
import cv2 as cv

import geometry
from chip_extract import extract_main_features
from contact_measurement import extract_contact_features, render_contact_features


def compute_chip_convex_hull(main_ft: MainFeatures, chip_pts: OpenCVIntArray) -> OpenCVIntArray:
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


# def extract_chip_curve_points(main_ft: MainFeatures, chip_hull_pts: OpenCVIntArray) -> OpenCVIntArray:
#     """Return the points of the chip hull which belong to the chip curve."""
#     return geometry.under_lines(
#         chip_hull_pts,
#         (main_ft.base_line, main_ft.base_opp_border, main_ft.tool_opp_border),
#         (0, 15, 15)
#     )


def geometrical_analysis(binary_img: GrayImage) -> tuple[MainFeatures, ContactFeatures, InsideFeatures]:
    main_ft = extract_main_features(binary_img)

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))
    chip_convex_hull_pts = compute_chip_convex_hull(main_ft, chip_pts)

    outside_segments = geometry.under_lines(
        chip_convex_hull_pts,
        (main_ft.base_line, main_ft.base_opp_border, main_ft.tool_opp_border),
        (-5, 15, 15)
    )

    contact_ft = extract_contact_features(main_ft, outside_segments)
    inside_ft = ...

    return main_ft, contact_ft, inside_ft
