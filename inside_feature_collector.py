from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable, Sequence
    from shape_detection.chip_extraction import MainFeatures
    from shape_detection.chip_inside_contour import InsideFeatures
    from pathlib import Path

import abc

import numpy as np

import csv
import matplotlib.pyplot as plt

import scipy.signal as scs
from filterpy.kalman import KalmanFilter  # pip install filterpy
import pywt  # pip install pywavelets

from shape_detection.chip_inside_contour import (
    extract_chip_inside_contour,
    render_inside_features
)
from graph_animation import GraphAnimation



def erase_down_spikes(signal: Sequence[float], kernel_size:int) -> Sequence[float]:
    assert kernel_size % 2 == 1, 'kernel_size must be odd'
    off = kernel_size // 2
    smoothed = []
    for i in range(len(signal)):
        values = sorted(signal[max(0, i-off):i+off+1])
        median = values[len(values) // 2]
        smoothed.append(max(signal[i], median))
    return smoothed


class AbstractInsideFeatureCollector(abc.ABC):
    xlabel = "inside contour point index"
    ylabel = "thickness along the chip (µm)"

    def __init__(self, scale: float=1.0):
        self.scale = scale
        self.inside_features: list[InsideFeatures] = []
        self.main_features: list[MainFeatures] = []

    def collect(self, main_ft: MainFeatures, inside_ft: InsideFeatures) -> None:
        self.main_features.append(main_ft)
        self.inside_features.append(inside_ft)

    def extract_and_render(self, binary_img: np.ndarray, background: np.ndarray|None=None) -> np.ndarray:
        main_ft, inside_ft = extract_chip_inside_contour(binary_img)
        self.collect(main_ft, inside_ft)
        if background is None:
            ft_repr = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
            render_inside_features(ft_repr, main_ft, inside_ft)
            # ft_repr[binary_img > 0] = (255, 255, 255)
        else:
            ft_repr = background.copy()
            render_inside_features(ft_repr, main_ft, inside_ft)
        return ft_repr

    def show_thickness_graph(self, frame_index: int) -> None:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(True)
        for measure_sequences in self.get_measures():
            ax.plot(measure_sequences[frame_index], '-x')
        plt.show()

    def show_thickness_animated_graph(self) -> None:
        anim = GraphAnimation(self.get_measures(), self.xlabel, self.ylabel)
        anim.play()

    @abc.abstractmethod
    def get_measures(self) -> Sequence[list[Sequence[float]]]:
        pass

    def save_measures(self, save_file_path: Path, frame_index: int) -> None:
        if save_file_path.suffix.lower() != '.csv':
            raise ValueError("the save file must be a csv file")
        with save_file_path.open(mode='w') as csv_save_file:
            csv_writer = csv.writer(csv_save_file)
            for measure in self.get_measures():
                csv_writer.writerow(measure[frame_index])


class CollectorKalman(AbstractInsideFeatureCollector):
    def __init__(self, scale: float=1.0):
        super().__init__(scale)
        self.thickness_seqs: list[Sequence[float]] = []
        self.smoothed_seqs: list[Sequence[float]] = []

        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1, 1], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.R = np.array([[1]])
        kf.Q = np.array([[0.001, 0], [0, 0.001]])
        kf.P = np.array([[1, 0], [0, 1]])
        kf.x = np.array([[0], [0]])
        self.kf_filter = kf

    def get_measures(self) -> Sequence[list[Sequence[float]]]:
        return (self.thickness_seqs, self.smoothed_seqs)

    def collect(self, main_ft: MainFeatures, inside_ft: InsideFeatures) -> None:
        super().collect(main_ft, inside_ft)
        self.thickness_seqs.append(inside_ft.thickness)
        smoothed = []
        for z in inside_ft.thickness:
            self.kf_filter.predict()
            self.kf_filter.update([z])
            smoothed.append(self.kf_filter.x[0, 0])
        self.smoothed_seqs.append(smoothed)


class CollectorMedian(AbstractInsideFeatureCollector):
    def __init__(self, scale: float=1.0):
        super().__init__(scale)
        self.thickness_seqs: list[Sequence[float]] = []
        self.smoothed_seqs: list[Sequence[float]] = []
        self.extra_smoothed_seqs: list[Sequence[float]] = []

    def get_measures(self) -> Sequence[list[Sequence[float]]]:
        return (self.thickness_seqs, self.smoothed_seqs, self.extra_smoothed_seqs)

    def collect(self, main_ft: MainFeatures, inside_ft: InsideFeatures) -> None:
        super().collect(main_ft, inside_ft)

        signal = [self.scale * t for t in inside_ft.thickness]
        smoothed = erase_down_spikes(signal, kernel_size=5)
        extra_smoothed = scs.medfilt(smoothed, kernel_size=7)

        self.thickness_seqs.append(signal)
        self.smoothed_seqs.append(smoothed)
        self.extra_smoothed_seqs.append(extra_smoothed)



class CollectorWavelet(AbstractInsideFeatureCollector):
    def __init__(self, scale: float=1.0):
        super().__init__(scale)
        self.thickness_seqs: list[Sequence[float]] = []
        self.smoothed_seqs: list[Sequence[float]] = []

    def get_measures(self) -> Sequence[list[Sequence[float]]]:
        return (self.thickness_seqs, self.smoothed_seqs)

    def collect(self, main_ft: MainFeatures, inside_ft: InsideFeatures) -> None:
        super().collect(main_ft, inside_ft)

        signal = [self.scale * t for t in inside_ft.thickness]

        wavelet = 'db4'
        coeffs = pywt.wavedec(signal, wavelet, level=3)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = 10 * sigma * np.sqrt(2 * np.log(len(signal)))
        filtered_coeffs = [pywt.threshold(c, threshold, mode='hard') for c in coeffs]
        denoised_signal = pywt.waverec(filtered_coeffs, wavelet)

        self.thickness_seqs.append(signal)
        self.smoothed_seqs.append(denoised_signal)



# TODO:
class CollectorDerivative(AbstractInsideFeatureCollector):
    def __init__(self, scale: float=1.0):
        super().__init__(scale)
        self.thickness_seqs: list[Sequence[float]] = []
        ...

    def get_measures(self) -> Sequence[list[Sequence[float]]]:
        ...

    def collect(self, main_ft: MainFeatures, inside_ft: InsideFeatures) -> None:
        super().collect(main_ft, inside_ft)
