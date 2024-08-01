from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    from type_hints import GrayImage, OpenCVFloatArray, FloatPt, Line
    from features_main import MainFeatures

import numpy as np
import cv2 as cv

import geometry


class ToolTipFeatures:
    __slots__ = (
        "tool_tip_pt",  # type: FloatPt

        "tool_tip_line",   # type: Line
        "mean_tool_line",  # type: Line
        "mean_base_line"   # type: Line
    )


def best_tip_line(lines: OpenCVFloatArray) -> tuple[float, float]:
    high = 3*np.pi/4
    low = np.pi/4
    for rho, theta in lines[:, 0, :]:
        if low < theta < high:
            return rho, theta
    raise ValueError("tip line not found")


def locate_tool_tip(
    preprocessed_batch: Sequence[GrayImage],
    main_features: Sequence[MainFeatures]
) -> ToolTipFeatures:
    tip_ft = ToolTipFeatures()

    mean_bin_img = np.mean(preprocessed_batch, axis=0).astype(np.uint8)
    thresh = np.median(mean_bin_img[mean_bin_img > 100])
    cv.threshold(mean_bin_img, thresh, 255, cv.THRESH_BINARY, dst=mean_bin_img)

    tip_ft.mean_base_line = np.mean([main_ft.base_line for main_ft in main_features], axis=0)
    tip_ft.mean_tool_line = np.mean([main_ft.tool_line for main_ft in main_features], axis=0)

    y, x = np.nonzero(mean_bin_img)
    pts = np.column_stack((x, y))
    above_pts = geometry.above_lines(pts, (tip_ft.mean_base_line, tip_ft.mean_tool_line), (-5, -5))

    tip_bin_img = np.zeros_like(mean_bin_img)
    tip_bin_img[above_pts[:, 1], above_pts[:, 0]] = 255

    lines = cv.HoughLines(tip_bin_img, 1, np.pi/180, 1)
    if lines is None or len(lines) < 1:
        raise ValueError("tip line not found")

    rho_tip, theta_tip = geometry.standard_polar_param(*best_tip_line(lines))
    xn_tip, yn_tip = np.cos(theta_tip), np.sin(theta_tip)
    tip_ft.tool_tip_line = (rho_tip, xn_tip, yn_tip)

    tip_ft.tool_tip_pt = geometry.intersect_line(tip_ft.mean_tool_line, tip_ft.tool_tip_line)

    return tip_ft
