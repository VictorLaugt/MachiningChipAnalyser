from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable
    from type_hints import GrayImage, FloatPt, FloatArray, Line, OpenCVFloatArray
    from img_loader import AbstractImageLoader
    from geometrical_analysis import GeometricalFeatures
    from outputs_measurement_writer import AbstractMeasurementWriter
    from outputs_feature_renderer import AbstractFeatureRenderer
    from output_thickness_analysis_animator import AbstractThicknessAnalysisAnimator

import numpy as np
import cv2 as cv
from scipy.signal import savgol_filter, find_peaks

import geometry
from preproc import preprocess
from geometrical_analysis import extract_geometrical_features


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


def best_tip_line(lines: OpenCVFloatArray) -> tuple[float, float]:
    # MOCK: best_tip_line function
    from features_main import best_base_line
    return best_base_line(lines)


def locate_tip_line(tip_bin_img: GrayImage, base_line: Line, tool_line: Line) -> Line:
    y, x = np.nonzero(tip_bin_img)
    pts = np.column_stack((x, y)).reshape(-1, 1, 2)

    above_pts = geometry.above_lines(pts, (base_line, tool_line), (-5, -5))
    tip_bin_img = np.zeros_like(tip_bin_img)
    tip_bin_img[above_pts[:, 0, 1], above_pts[:, 0, 0]] = 255

    lines = cv.HoughLines(tip_bin_img, 1, np.pi/180, 1)
    if lines is None or len(lines) < 1:
        raise ValueError("tip line not found")

    rho_tip, theta_tip = geometry.standard_polar_param(*best_tip_line(lines))
    xn_tip, yn_tip = np.cos(theta_tip), np.sin(theta_tip)

    return rho_tip, xn_tip, yn_tip


def measure_contact_length(tool_base_inter: FloatPt, contact_point: FloatPt) -> float:
    xi, yi = tool_base_inter
    xc, yc = contact_point
    return np.linalg.norm((xc-xi, yc-yi))


def measure_spike_valley_thickness(
    tool_base_inter: FloatPt,
    tool_tip: FloatPt,
    thickness: FloatArray
) -> ThicknessAnalysis:
    an = ThicknessAnalysis()

    xi, yi = tool_base_inter
    xt, yt = tool_tip
    tool_penetration = np.linalg.norm((xt-xi, yt-yi))

    # TODO: use tool_penetration to compute window_length and prominence
    an.rough_thk = savgol_filter(thickness, window_length=45, polyorder=2)
    an.smoothed_thk = savgol_filter(thickness, window_length=15, polyorder=2)

    an.rough_spike_indices, _ = find_peaks(an.rough_thk, prominence=5)
    rough_period = np.mean(np.diff(an.rough_spike_indices))

    an.spike_indices, _ = find_peaks(an.smoothed_thk, distance=0.7*rough_period)
    an.valley_indices, _ = find_peaks(-an.smoothed_thk, distance=0.7*rough_period, width=0.2*rough_period)

    an.mean_spike_thickness = np.mean(an.smoothed_thk[an.spike_indices])
    an.mean_valley_thickness = np.mean(an.smoothed_thk[an.valley_indices])

    return an


class FeatureAccumulator:
    def __init__(
        self,
        img_shape: tuple[int, int],
        measurement_writer: AbstractMeasurementWriter,
        thickness_analysis_animator: AbstractThicknessAnalysisAnimator
    ) -> None:
        self.measurement_writer = measurement_writer
        self.thickness_analysis_animator = thickness_analysis_animator

        self.bin_img_sum = np.zeros(img_shape, dtype=np.int64)

        self.tool_base_inter: list[FloatPt] = []
        self.contact_point: list[FloatPt] = []

        self.base_line: list[Line] = []
        self.tool_line: list[Line] = []

        self.thickness: list[FloatArray] = []

    def accumulate(self, binary_img: GrayImage, features: GeometricalFeatures) -> None:
        self.bin_img_sum += binary_img

        self.tool_base_inter.append(features.main_ft.tool_base_intersection)
        self.contact_point.append(features.contact_ft.contact_point)

        self.base_line.append(features.main_ft.base_line)
        self.tool_line.append(features.main_ft.tool_line)

        self.thickness.append(features.inside_ft.thickness)

    def measure(self) -> None:
        img_nb = len(self.tool_base_inter)

        # compute the mean of the preprocessed images
        mean_bin_img = (self.bin_img_sum / img_nb).astype(np.uint8)
        thresh = np.median(mean_bin_img[mean_bin_img > 100])
        cv.threshold(mean_bin_img, thresh, 255, cv.THRESH_BINARY, dst=mean_bin_img)

        # locate the tool tip point
        mean_base_line = np.mean(self.base_line, axis=0)
        mean_tool_line = np.mean(self.tool_line, axis=0)
        tip_line = locate_tip_line(mean_bin_img, mean_base_line, mean_tool_line)
        tool_tip = geometry.intersect_line(mean_tool_line, tip_line)

        # measure chip characteristics
        for tool_base_inter, contact_point, thk in zip(
            self.tool_base_inter, self.contact_point, self.thickness
        ):
            contact_len = measure_contact_length(tool_base_inter, contact_point)
            thk_an = measure_spike_valley_thickness(tool_base_inter, tool_tip, thk)
            self.measurement_writer.write(contact_len, thk_an)
            self.thickness_analysis_animator.add_frame(thk, thk_an)


def analysis_loop(
    loader: AbstractImageLoader,
    measurement_writer: AbstractMeasurementWriter,
    feature_renderer: AbstractFeatureRenderer,
    thickness_analysis_animator: AbstractThicknessAnalysisAnimator,
    progress: Callable[[int, int, int], None]
) -> None:
    accumulator = FeatureAccumulator(
        loader.img_shape()[:2],
        measurement_writer,
        thickness_analysis_animator
    )
    img_nb = len(loader)

    for i, input_img in enumerate(loader):
        progress(i, img_nb, 10)

        binary_img = preprocess(input_img)
        features = extract_geometrical_features(binary_img)
        accumulator.accumulate(binary_img, features)

        feature_renderer.render_frame(i, input_img, binary_img, features)

    accumulator.measure()
