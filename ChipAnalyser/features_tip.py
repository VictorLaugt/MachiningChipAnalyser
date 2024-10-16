from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Optional
    from type_hints import GrayImage, OpenCVFloatArray
    from features_main import MainFeatures

import warnings

import numpy as np
import cv2 as cv

import geometry


class ToolTipFeatures:
    __slots__ = (
        "tool_tip_pt",  # type: IntPt

        "tool_tip_line",   # type: Line
        "mean_tool_line",  # type: Line
        "mean_base_line"   # type: Line
    )


def tool_tip_not_found_warning() -> None:
    warnings.warn("tool tip not found")


def best_tip_line(lines: OpenCVFloatArray) -> tuple[float, float]:
    """Among the lines detected by the Hough transform, return the one that best
    fits the tool tip.

    Parameter
    ---------
    line: (n, 1, 2)-array of float
        Array containing (rho, theta) polar parameters of the lines detected by
        the Hough transform, sorted in descending order of received votes.

    Returns
    -------
    best_line: float couple
        (rho, theta) polar parameters of the best line to fit the tool tip.
    """
    high = 3*np.pi/4
    low = np.pi/4
    for rho, theta in lines[:, 0, :]:
        if low < theta < high:
            return rho, theta
    return np.nan, np.nan


def locate_tool_tip(
    preprocessed_batch: Sequence[GrayImage],
    main_features: Sequence[MainFeatures]
) -> Optional[ToolTipFeatures]:
    """Compute the position of the tool tip point in the average of all the
    batch's preprocessed machining images.

    Parameters
    ----------
    preprocessed_batch: sequence of (h, w)-arrays of uint8
        Every preprocessed machining images of the batch.
    main_features: sequence of MainFeatures
        MainFeatures structures corresponding to each machining image of the batch.

    Returns
    -------
    tip_ft: ToolTipFeatures or None if the extraction fails
        Structure containing tip features, including the tool tip point.
    """
    tip_ft = ToolTipFeatures()

    mean_bin_img = np.mean(preprocessed_batch, axis=0).astype(np.uint8)
    thresh = np.median(mean_bin_img[mean_bin_img > 100])
    cv.threshold(mean_bin_img, thresh, 255, cv.THRESH_BINARY, dst=mean_bin_img)

    base_lines = [main_ft.base_line for main_ft in main_features if main_ft is not None]
    tool_lines = [main_ft.tool_line for main_ft in main_features if main_ft is not None]
    if len(base_lines) == 0:
        return tool_tip_not_found_warning()

    tip_ft.mean_base_line = np.mean(base_lines, axis=0)
    tip_ft.mean_tool_line = np.mean(tool_lines, axis=0)

    y, x = np.nonzero(mean_bin_img)
    pts = np.column_stack((x, y))
    above_pts = geometry.above_lines(pts, (tip_ft.mean_base_line, tip_ft.mean_tool_line), (-5, -5))

    tip_bin_img = np.zeros_like(mean_bin_img)
    tip_bin_img[above_pts[:, 1], above_pts[:, 0]] = 255

    lines = cv.HoughLines(tip_bin_img, 1, np.pi/180, 1)
    if lines is None:
        return tool_tip_not_found_warning()


    rho_tip, theta_tip = geometry.standard_polar_param(*best_tip_line(lines))
    if theta_tip is None:
        return tool_tip_not_found_warning()

    xn_tip, yn_tip = np.cos(theta_tip), np.sin(theta_tip)
    tip_ft.tool_tip_line = (rho_tip, xn_tip, yn_tip)
    tip_ft.tool_tip_pt = geometry.intersect_line_safe(tip_ft.mean_tool_line, tip_ft.tool_tip_line)
    if tip_ft.tool_tip_pt is None:
        return tool_tip_not_found_warning()

    return tip_ft
