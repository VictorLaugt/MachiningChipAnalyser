from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from type_hints import GrayImage, OpenCVFloatArray, Line

import warnings

import numpy as np
import cv2 as cv

import geometry


class MainFeatures:
    __slots__ = (
        "indirect_rotation",  # type: bool
        "base_line",          # type: Line
        "tool_line",          # type: Line

        "tool_angle",  # type: float

        "base_border",      # type: Line
        "base_opp_border",  # type: Line
        "tool_opp_border",  # type: Line

        "tool_base_inter_pt",  # type: IntPt
    )


def best_base_line(lines: OpenCVFloatArray) -> tuple[float, float]:
    """Among the lines detected by the Hough transform, return the one that best
    fits the base of the part being machined.

    Parameter
    ---------
    line: (n, 1, 2)-array of float
        Array containing (rho, theta) polar parameters of the lines detected by
        the Hough transform, sorted in descending order of received votes.

    Returns
    -------
    best_line: float couple
        (rho, theta) polar parameters of the best line to fit the base.
    """
    for rho, theta in lines[:, 0, :]:
        if np.abs(theta - np.pi/2) < 0.2:
            return rho, theta
    warnings.warn("base line not found")
    return np.nan, np.nan


def best_tool_line(lines: OpenCVFloatArray) -> tuple[float, float]:
    """Among the lines detected by the Hough transform, return the one that best
    fits the blade of the tool.

    Parameter
    ---------
    line: (n, 1, 2)-array of float
        Array containing (rho, theta) polar parameters of the lines detected by
        the Hough transform, sorted in descending order of received votes.

    Returns
    -------
    best_line: float couple
        (rho, theta) polar parameters of the best line to fit the tool.
    """
    high = np.pi/8
    low = 7*np.pi/8
    for rho, theta in lines[:, 0, :]:
        if theta < high or theta > low:
            return rho, theta
    warnings.warn("tool line not found")
    return np.nan, np.nan


def locate_base_and_tool(binary_img: GrayImage) -> tuple[Line, Line, float, float]:
    """Compute two lines that fit the base of the part being machined and the
    blade of the tool.

    The lines are returned as (rho, xn, yn) triplets.
    (rho, theta) are the polar parameters of the line and (xn, yn) is the unit
    normal vector of the line, which means xn = cos(theta) and yn = sin(theta).

    Parameter
    ---------
    binary_img: (h, w)-array of uint8
        Preprocessed machining image.

    Returns
    -------
    base_line: float triplet
        Line fitting the base.
    tool_line: float triplet
        Line fitting the tool.
    theta_base: float
        Angle between base_line and the y-axis.
    theta_tool: float
        Angle between tool_line and the y-axis.
    """
    lines = cv.HoughLines(binary_img, 1, np.pi/180, 100)
    if lines is None:
        warnings.warn("no line found")
        return geometry.NAN_LINE, geometry.NAN_LINE, np.nan, np.nan

    rho_base, theta_base = geometry.standard_polar_param(*best_base_line(lines))
    rho_tool, theta_tool = geometry.standard_polar_param(*best_tool_line(lines))

    xn_base, yn_base = np.cos(theta_base), np.sin(theta_base)
    xn_tool, yn_tool = np.cos(theta_tool), np.sin(theta_tool)

    return (rho_base, xn_base, yn_base), (rho_tool, xn_tool, yn_tool), theta_base, theta_tool


def extract_main_features(binary_img: GrayImage) -> Optional[MainFeatures]:
    """Extract the main features of a preprocessed machining image.

    The parameterization of base_line and tool_line is such that the chip pixels
    are those located below these two lines, i.e. on the opposite side of their
    normal vector (xn, yn).

    Parameter
    ---------
    binary_img: (h, w)-array of uint8
        Preprocessed machining image.

    Returns
    -------
    main_ft: MainFeatures or None if the extraction fails
        Structure containing main features of the image, including:
        - base_line: a line that fits the base of the part being machined
        - tool_line: a line that fits the blade of the tool
        - borders of the image relative base_line and tool_line
    """
    ft = MainFeatures()
    h, w = binary_img.shape

    base_line, tool_line, theta_base, theta_tool = locate_base_and_tool(binary_img)
    if np.isnan(theta_base) or np.isnan(theta_tool):
        return

    ft.tool_angle = theta_tool
    ft.tool_base_inter_pt = (xi, yi) = geometry.intersect_line(base_line, tool_line)

    _, xn_base, yn_base = base_line
    _, xn_tool, yn_tool = tool_line
    direct_base_tool = xn_base*yn_tool - yn_base*xn_tool > 0
    up, left, down, right = (0, 0, -1), (0, -1, 0), (h, 0, 1), (w, 1, 0)

    if xi > w/2:
        # down right intersection
        if yi > h/2:
            ft.indirect_rotation = not direct_base_tool
            ft.base_line, ft.tool_line = base_line, tool_line
            if direct_base_tool:
                ft.base_border, ft.base_opp_border, ft.tool_opp_border = right, left, up
            else:
                ft.base_border, ft.base_opp_border, ft.tool_opp_border = down, up, left

        # up right intersection
        else:
            ft.indirect_rotation = direct_base_tool
            if direct_base_tool:
                ft.base_line, ft.tool_line = base_line, geometry.neg_line(tool_line)
                ft.base_border, ft.base_opp_border, ft.tool_opp_border = right, left, down
            else:
                ft.base_line, ft.tool_line = geometry.neg_line(base_line), tool_line
                ft.base_border, ft.base_opp_border, ft.tool_opp_border = up, down, left

    else:
        # down left intersection
        if yi > h/2:
            ft.indirect_rotation = direct_base_tool
            if direct_base_tool:
                ft.base_line, ft.tool_line = geometry.neg_line(base_line), tool_line
                ft.base_border, ft.base_opp_border, ft.tool_opp_border = left, right, up
            else:
                ft.base_line, ft.tool_line = base_line, geometry.neg_line(tool_line)
                ft.base_border, ft.base_opp_border, ft.tool_opp_border = down, up, right

        # up left intersection
        else:
            ft.indirect_rotation = not direct_base_tool
            ft.base_line, ft.tool_line = geometry.neg_line(base_line), geometry.neg_line(tool_line)
            if direct_base_tool:
                ft.base_border, ft.base_opp_border, ft.tool_opp_border = left, right, down
            else:
                ft.base_border, ft.base_opp_border, ft.tool_opp_border = up, down, right

    return ft
