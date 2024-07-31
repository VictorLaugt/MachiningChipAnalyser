from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from type_hints import ColorImage, IntPtArray
    from measure import ToolTipFeatures
    from features_main import MainFeatures

import sys
import matplotlib.pyplot as plt

import geometry
import colors

import numpy as np
from numpy.polynomial import Polynomial
import cv2 as cv


class ContactFeatures:
    __slots__ = (
        "contact_pt",  # type: FloatPt
        "key_pts",     # type: IntPtArray
        "polynomial"   # type: Polynomial
    )


def extract_key_points(main_ft: MainFeatures, curve_points: IntPtArray, tool_chip_max_angle: float) -> IntPtArray:
    """Return key points from the chip curve which can then be use to fit a polynomial."""
    if main_ft.tool_angle > np.pi:
        tool_angle = main_ft.tool_angle - 2*np.pi
    else:
        tool_angle = main_ft.tool_angle

    if main_ft.indirect_rotation:
        curve_surface_vectors = curve_points[:-1, :] - curve_points[1:, :]
    else:
        curve_surface_vectors = curve_points[1:, :] - curve_points[:-1, :]

    curve_segment_lengths = np.linalg.norm(curve_surface_vectors, axis=-1)
    curve_vector_angles = np.arccos(curve_surface_vectors[:, 0] / curve_segment_lengths)

    mask = np.ones(len(curve_points), dtype=bool)
    mask[1:] = np.abs(np.pi/2 + tool_angle - curve_vector_angles) < tool_chip_max_angle

    return curve_points[mask]


def fit_polynomial(main_ft: MainFeatures, key_pts: IntPtArray) -> Polynomial:
    """Return a polynomial of degree 2 which fits the key points."""
    rot_x, rot_y = geometry.rotate(key_pts[:, 0], key_pts[:, 1], -main_ft.tool_angle)

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
    xi, yi = main_ft.tool_base_inter_pt
    cos, sin = np.cos(main_ft.tool_angle), -np.sin(main_ft.tool_angle)
    rot_xc = xi*cos - yi*sin

    # ordinate of the contact point, rotated in the polynomial basis
    rot_yc = polynomial(rot_xc)

    # contact point, rotated back in the image basis
    return geometry.rotate(rot_xc, rot_yc, main_ft.tool_angle)


def extract_contact_features(main_ft: MainFeatures, outside_segments: IntPtArray) -> ContactFeatures:
    contact_ft = ContactFeatures()

    contact_ft.key_pts = extract_key_points(main_ft, outside_segments[1:], np.pi/4)
    contact_ft.polynomial = fit_polynomial(main_ft, contact_ft.key_pts)
    contact_ft.contact_pt = chip_tool_contact_point(main_ft, contact_ft.polynomial)

    return contact_ft


def render_contact_features(
    frame_num: int,
    render: ColorImage,
    main_ft: MainFeatures,
    tip_ft: ToolTipFeatures,
    contact_ft: ContactFeatures
) -> None:
    # draw polynomial approximation of the chip outside curve
    if contact_ft.polynomial is not None:
        x = np.arange(0, render.shape[1], 1, dtype=np.int32)
        y = contact_ft.polynomial(x)
        x, y = geometry.rotate(x, y, main_ft.tool_angle)
        x, y = x.astype(np.int32), y.astype(np.int32)
        for i in range(len(x)-1):
            cv.line(render, (x[i], y[i]), (x[i+1], y[i+1]), colors.BLUE, thickness=2)

    # draw detected lines for the tool, the base, and the tool tip
    geometry.draw_line(render, main_ft.tool_line, colors.RED, thickness=3)
    geometry.draw_line(render, main_ft.base_line, colors.RED, thickness=3)
    geometry.draw_line(render, tip_ft.tool_tip_line, colors.RED, thickness=3)

    # draw the points used for the polynomial fitting
    for kpt in contact_ft.key_pts:
        cv.circle(render, kpt, 6, colors.GREEN, thickness=-1)

    # draw the contact length
    xc, yc = contact_ft.contact_pt
    xt, yt = tip_ft.tool_tip_pt
    _, dx, dy = main_ft.tool_line
    cv.line(render, (int(xt-50*dx), int(yt-50*dy)), (int(xt), int(yt)), colors.YELLOW, 1)
    cv.line(render, (int(xc-50*dx), int(yc-50*dy)), (int(xc), int(yc)), colors.YELLOW, 1)
    cv.arrowedLine(render, (int(xt-33*dx), int(yt-33*dy)), (int(xc-33*dx), int(yc-33*dy)), colors.YELLOW, 2)

    # write the frame number
    cv.putText(render, f"frame: {frame_num}", (20, render.shape[0]-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors.WHITE)
