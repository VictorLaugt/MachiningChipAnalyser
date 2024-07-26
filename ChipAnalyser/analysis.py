from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from type_hints import GrayImage, OpenCVIntArray
    from features_main import MainFeatures
    from features_contact import ContactFeatures
    from features_thickness import InsideFeatures

import numpy as np
import cv2 as cv

import geometry
from features_main import extract_main_features
from features_contact import extract_contact_features, measure_contact_length, render_contact_features
from features_thickness import extract_inside_features, measure_spike_valley_thickness, render_inside_features


class GeometricalFeatures:
    __slots__ = (
        'main_ft',     # type: MainFeatures
        'contact_ft',  # type: ContactFeatures
        'inside_ft'    # type: InsideFeatures
    )


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


def extract_geometrical_features(binary_img: GrayImage) -> GeometricalFeatures:
    geometrical_ft = GeometricalFeatures()

    main_ft = extract_main_features(binary_img)

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))

    chip_binary_img = np.zeros_like(binary_img)
    chip_binary_img[chip_pts[:, 0, 1], chip_pts[:, 0, 0]] = 255

    chip_convex_hull_pts = compute_chip_convex_hull(main_ft, chip_pts)
    outside_segments = geometry.under_lines(
        chip_convex_hull_pts,
        (main_ft.base_line, main_ft.base_opp_border, main_ft.tool_opp_border),
        (-5, 15, 15)
    )

    geometrical_ft.main_ft = main_ft
    geometrical_ft.contact_ft = extract_contact_features(main_ft, outside_segments)
    geometrical_ft.inside_ft = extract_inside_features(main_ft, outside_segments, chip_binary_img)

    return geometrical_ft
