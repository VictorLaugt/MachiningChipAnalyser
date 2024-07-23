from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geometry import Line, PointArray
    from typing import TypeVar
    PolarParamArray = TypeVar('PolarParamArray', np.ndarray)  # ~ (n, 1, 2) dtype=float32

from dataclasses import dataclass

import numpy as np
import cv2 as cv

import geometry


# class MainFeatures:
#     __slots__ = (
#         "indirect_rotation",       # type: bool
#         "base_line",               # type: Line
#         "tool_line",               # type: Line
#         "tool_angle",              # type: float
#         "base_border",             # type: Line
#         "base_opp_border",         # type: Line
#         "tool_opp_border",         # type: Line
#         "tool_base_intersection",  # type: tuple[float, float]
#     )


@dataclass
class MainFeatures:
    def __init__(self):
        pass

    # rotation direction of the chip
    indirect_rotation: bool

    # base and tool lines, such that every points of the chip are under these lines
    base_line: Line
    tool_line: Line

    # angle of the tool line polar coordinates
    tool_angle: float

    # borders of the image
    base_border: Line
    base_opp_border: Line
    tool_opp_border: Line

    # intersection between the base and the tool lines
    tool_base_intersection: tuple[float, float]


def best_base_line(lines: PolarParamArray) -> tuple[float, float]:
    """Return the best horizontal line."""
    for rho, theta in lines[:, 0, :]:
        if np.abs(theta - np.pi/2) < 0.2:
            return rho, theta
    raise ValueError("base not found")


def best_tool_line(lines: PolarParamArray) -> tuple[float, float]:
    """Return the best tool line."""
    high = np.pi/8
    low = 7*np.pi/8
    for rho, theta in lines[:, 0, :]:
        if theta < high or theta > low:
            return rho, theta
    raise ValueError("tool not found")


def locate_base_and_tool(binary_img: np.ndarray) -> tuple[Line, Line, float, float]:
    """Compute line parameters for base and tool."""
    lines = cv.HoughLines(binary_img, 1, np.pi/180, 100)
    if lines is None or len(lines) < 2:
        raise ValueError("line not found")

    rho_base, theta_base = geometry.standard_polar_param(*best_base_line(lines))
    rho_tool, theta_tool = geometry.standard_polar_param(*best_tool_line(lines))

    xn_base, yn_base = np.cos(theta_base), np.sin(theta_base)
    xn_tool, yn_tool = np.cos(theta_tool), np.sin(theta_tool)

    return (rho_base, xn_base, yn_base), (rho_tool, xn_tool, yn_tool), theta_base, theta_tool


def extract_main_features(binary_img: np.ndarray) -> MainFeatures:
    ft = MainFeatures()
    h, w = binary_img.shape

    base_line, tool_line, _, ft.tool_angle = locate_base_and_tool(binary_img)
    ft.tool_base_intersection = xi, yi = geometry.intersect_line(base_line, tool_line)

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


def render_main_features(binary_img: np.ndarray) -> np.ndarray:
    main_ft = extract_main_features(binary_img)

    red = (0, 0, 255)
    white = (255, 255, 255)
    render = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)

    geometry.draw_line(render, main_ft.base_line, color=red, thickness=3)
    geometry.draw_line(render, main_ft.tool_line, color=red, thickness=3)

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))
    render[chip_pts[:, 0, 1], chip_pts[:, 0, 0]] = white

    return render


if __name__ == '__main__':
    import os
    from pathlib import Path

    import image_loader
    import preprocessing.log_tresh_blobfilter_erode

    # ---- environment variables
    input_dir_str = os.environ.get("INPUT_DIR")
    output_dir_str = os.environ.get("OUTPUT_DIR")
    scale_str = os.environ.get("SCALE_UM")

    if input_dir_str is not None:
        input_dir = Path(input_dir_str)
    else:
        input_dir = Path("imgs", "vertical")

    if output_dir_str is not None:
        output_dir = Path(output_dir_str)
    else:
        output_dir = Path("results", "chipvurve")

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("chipextraction", render_main_features)

    # ---- processing
    loader = image_loader.ImageLoader(input_dir)
    processing.run(loader)

    # ---- visualization
    processing.show_frames(min(14, len(loader)-1))
    # processing.show_video_comp(("input", "chipextraction"))

    processing.show_frame_comp(min(14, len(loader)-1), ("morph", "chipextraction"))
    processing.save_frame_comp(output_dir, 14, ("morph",))
    processing.save_frame_comp(output_dir, 14, ("chipextraction",))
    # processing.save_videos(output_dir)
