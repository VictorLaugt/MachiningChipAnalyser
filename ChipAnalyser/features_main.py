from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from type_hints import GrayImage, OpenCVFloatArray, Line

from dataclasses import dataclass

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

        "tool_base_inter_pt",  # type: FloatPt
    )


def best_base_line(lines: OpenCVFloatArray) -> tuple[float, float]:
    """Return the best horizontal line."""
    for rho, theta in lines[:, 0, :]:
        if np.abs(theta - np.pi/2) < 0.2:
            return rho, theta
    raise ValueError("base not found")


def best_tool_line(lines: OpenCVFloatArray) -> tuple[float, float]:
    """Return the best tool line."""
    high = np.pi/8
    low = 7*np.pi/8
    for rho, theta in lines[:, 0, :]:
        if theta < high or theta > low:
            return rho, theta
    raise ValueError("tool not found")


def locate_base_and_tool(binary_img: GrayImage) -> tuple[Line, Line, float, float]:
    """Compute line parameters for base and tool."""
    lines = cv.HoughLines(binary_img, 1, np.pi/180, 100)
    if lines is None or len(lines) < 2:
        raise ValueError("line not found")

    rho_base, theta_base = geometry.standard_polar_param(*best_base_line(lines))
    rho_tool, theta_tool = geometry.standard_polar_param(*best_tool_line(lines))

    xn_base, yn_base = np.cos(theta_base), np.sin(theta_base)
    xn_tool, yn_tool = np.cos(theta_tool), np.sin(theta_tool)

    return (rho_base, xn_base, yn_base), (rho_tool, xn_tool, yn_tool), theta_base, theta_tool


def extract_main_features(binary_img: GrayImage) -> MainFeatures:
    ft = MainFeatures()
    h, w = binary_img.shape

    base_line, tool_line, _, ft.tool_angle = locate_base_and_tool(binary_img)
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
