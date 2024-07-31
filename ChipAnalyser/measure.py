from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    from type_hints import Image, GrayImage, Line, FloatPt, FloatArray, OpenCVFloatArray
    from features_main import MainFeatures
    from outputs_measurement_writer import AbstractMeasurementWriter
    from outputs_analysis_renderer import AbstractAnalysisRenderer

import numpy as np
import cv2 as cv
from scipy.signal import savgol_filter, find_peaks

import geometry
from preproc import preprocess
from features_main import extract_main_features
from chip_analysis import extract_chip_features


class ToolTipFeatures:
    __slots__ = (
        "tool_tip_pt",  # type: FloatPt

        "tool_tip_line",   # type: Line
        "mean_tool_line",  # type: Line
        "mean_base_line"   # type: Line
    )


class ThicknessAnalysis:
    __slots__ = (
        "rough_thk",     # type: FloatArray
        "smoothed_thk",  # type: FloatArray

        "rough_spike_indices",  # type: IntArray
        "spike_indices",        # type: IntArray
        "valley_indices",       # type: IntArray

        "mean_spike_thickness",   # type: float
        "mean_valley_thickness",  # type: float
    )


def pt2pt_distance(pt0: FloatPt, pt1: FloatPt) -> float:
    (x0, y0), (x1, y1) = pt0, pt1
    return np.linalg.norm((x1-x0, y1-y0))


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
    pts = np.column_stack((x, y)).reshape(-1, 1, 2)
    above_pts = geometry.above_lines(pts, (tip_ft.mean_base_line, tip_ft.mean_tool_line), (-5, -5))

    tip_bin_img = np.zeros_like(mean_bin_img)
    tip_bin_img[above_pts[:, 0, 1], above_pts[:, 0, 0]] = 255

    lines = cv.HoughLines(tip_bin_img, 1, np.pi/180, 1)
    if lines is None or len(lines) < 1:
        raise ValueError("tip line not found")

    rho_tip, theta_tip = geometry.standard_polar_param(*best_tip_line(lines))
    xn_tip, yn_tip = np.cos(theta_tip), np.sin(theta_tip)
    tip_ft.tool_tip_line = (rho_tip, xn_tip, yn_tip)

    tip_ft.tool_tip_pt = geometry.intersect_line(tip_ft.mean_tool_line, tip_ft.tool_tip_line)

    return tip_ft


def measure_contact_length(tool_tip_pt: FloatPt, contact_pt: FloatPt) -> float:
    xi, yi = tool_tip_pt
    xc, yc = contact_pt
    return np.linalg.norm((xc-xi, yc-yi))


def measure_spike_valley_thickness(
    thickness: FloatArray,
    tool_penetration: float
) -> ThicknessAnalysis:
    an = ThicknessAnalysis()

    an.rough_thk = savgol_filter(thickness, window_length=45, polyorder=2)
    an.smoothed_thk = savgol_filter(thickness, window_length=15, polyorder=2)

    an.rough_spike_indices, _ = find_peaks(an.rough_thk, prominence=0.08*tool_penetration)
    rough_period = np.mean(np.diff(an.rough_spike_indices))

    an.spike_indices, _ = find_peaks(an.smoothed_thk, distance=0.7*rough_period)
    an.valley_indices, _ = find_peaks(-an.smoothed_thk, distance=0.7*rough_period, width=0.2*rough_period)

    an.mean_spike_thickness = np.mean(an.smoothed_thk[an.spike_indices])
    an.mean_valley_thickness = np.mean(an.smoothed_thk[an.valley_indices])

    return an


def measure_characteristics(
    input_batch: Sequence[Image],
    measurement_writer: AbstractMeasurementWriter,
    analysis_renderer: AbstractAnalysisRenderer,
) -> None:
    preprocessed_batch: list[GrayImage] = []
    main_features: list[MainFeatures] = []
    for input_img in input_batch:
        binary_img = preprocess(input_img)
        main_ft = extract_main_features(binary_img)

        preprocessed_batch.append(binary_img)
        main_features.append(main_ft)

    tip_ft = locate_tool_tip(preprocessed_batch, main_features)

    for input_img, binary_img, main_ft in zip(input_batch, preprocessed_batch, main_features):
        tool_penetration = pt2pt_distance(tip_ft.tool_tip_pt, main_ft.tool_base_inter_pt)
        chip_ft = extract_chip_features(binary_img, main_ft, tool_penetration)

        contact_len = pt2pt_distance(tip_ft.tool_tip_pt, chip_ft.contact_ft.contact_pt)
        thickness_analysis = measure_spike_valley_thickness(chip_ft.inside_ft.thickness, tool_penetration)

        measurement_writer.write(contact_len, thickness_analysis)
        analysis_renderer.render_frame(input_img, binary_img, main_ft, tip_ft, chip_ft, thickness_analysis)
