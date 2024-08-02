from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from type_hints import GrayImage, IntPtArray
    from features_main import MainFeatures

import numpy as np
import cv2 as cv

import geometry
from features_contact import extract_contact_features
from features_thickness import extract_inside_features


class ChipFeatures:
    __slots__ = (
        'contact_ft',  # type: ContactFeatures
        'inside_ft'    # type: InsideFeatures
    )


def compute_chip_convex_hull(main_ft: MainFeatures, chip_pts: IntPtArray) -> IntPtArray:
    """Compute the convex hull of the chip while constraining it to pass through
    three anchor points.

    Parameters
    ----------
    main_ft: MainFeatures
        Main features of the preprocessed machining image.
    chip_pts: (n, 2)-array of int
        Points which belong to the chip

    Returns
    -------
    convex_hull_pts: (m, 2)-array of int
        Convex hull points sorted in the chip rotation order. The first point is
        the intersection between the tool line and the base line. m < n.
    """
    highest_idx, _ = geometry.line_furthest_point(chip_pts, main_ft.base_line)
    chip_highest = chip_pts[highest_idx, :]

    anchor_0 = geometry.orthogonal_projection(*chip_highest, main_ft.tool_opp_border)
    anchor_1 = geometry.orthogonal_projection(*anchor_0, main_ft.base_border)
    anchor_2 = main_ft.tool_base_inter_pt
    anchors = np.array([anchor_0, anchor_1, anchor_2], dtype=np.int32)

    chip_hull_pts = cv.convexHull(
        np.vstack((chip_pts, anchors)),
        clockwise=main_ft.indirect_rotation
    ).reshape(-1, 2)

    first_pt_idx = np.where(
        (chip_hull_pts[:, 0] == anchor_2[0]) &
        (chip_hull_pts[:, 1] == anchor_2[1])
    )[0][0]

    return np.roll(chip_hull_pts, -first_pt_idx, axis=0)


def extract_chip_features(binary_img: GrayImage, main_ft: MainFeatures, tool_penetration: float) -> ChipFeatures:
    """Extract the features of the chip on the preprocessed machining image.

    These features are divided into two categories:
    - those concerning the contact between the chip and the tool
    - those concerning the inside contour of the chip (i.e. the non-convex
    section of the chip contour)

    Parameters
    ----------
    binary_img: (h, w)-array of uint8
        Preprocessed machining image.
    main_ft: MainFeatures
        Main features of the preprocessed machining image.
    tool_penetration: float
        Tool penetration length into the part being machined.

    Returns
    -------
    chip_ft: ChipFeatures
        Structure containing features of the chip.
    """
    chip_ft = ChipFeatures()

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours).reshape(-1, 2)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))

    chip_binary_img = np.zeros_like(binary_img)
    chip_binary_img[chip_pts[:, 1], chip_pts[:, 0]] = 255

    chip_convex_hull_pts = compute_chip_convex_hull(main_ft, chip_pts)
    out_curve = geometry.under_lines(
        chip_convex_hull_pts,
        (main_ft.base_line, main_ft.base_opp_border, main_ft.tool_opp_border),
        (-5, 15, 15)
    )

    chip_ft.contact_ft = extract_contact_features(main_ft, out_curve)
    chip_ft.inside_ft = extract_inside_features(main_ft, out_curve, chip_binary_img, tool_penetration)

    return chip_ft
