from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from type_hints import ColorImage, GrayImage, FloatArray, FloatPtArray
    from pathlib import Path
    from measure import ToolTipFeatures
    from features_main import MainFeatures
    from chip_analysis import ChipFeatures
    from features_thickness import ThicknessAnalysis

import abc
import cv2 as cv
import numpy as np

from outputs_graph_animations import GraphAnimator

from features_contact import render_contact_features
from features_thickness import render_inside_features


EMPTY_FLOAT_ARRAY: FloatArray = np.empty((0,), dtype=float)
EMPTY_FLOAT_PT_ARRAY: FloatPtArray = np.empty((0, 2), dtype=float)


class AbstractAnalysisRenderer(abc.ABC):
    @abc.abstractmethod
    def render_frame(
        self,
        input_img: ColorImage,
        preproc_img: GrayImage,
        main_ft: MainFeatures,
        tip_ft: ToolTipFeatures,
        chip_ft: ChipFeatures,
        thickness_analysis: ThicknessAnalysis
    ) -> None:
        pass

    @abc.abstractmethod
    def no_render(self, input_img: ColorImage, preproc_img: GrayImage) -> None:
        pass

    @abc.abstractmethod
    def release(self) -> None:
        pass

    def __enter__(self) -> AbstractAnalysisRenderer:
        return self

    def __exit__(self, _exc_type, _exc_value, _exc_traceback) -> None:
        self.release()


class NoRendering(AbstractAnalysisRenderer):
    def render_frame(
        self,
        _input_img: ColorImage,
        _preproc_img: GrayImage,
        _main_ft: MainFeatures,
        _tip_ft: ToolTipFeatures,
        _chip_ft: ChipFeatures,
        _thickness_analysis: ThicknessAnalysis
    ) -> None:
        return

    def no_render(self, _input_img: ColorImage, _preproc_img: GrayImage) -> None:
        return

    def release(self) -> None:
        return


class AnalysisRenderer(AbstractAnalysisRenderer):
    def __init__(self, output_dir: Path, scale: float, image_height: int, image_width: int) -> None:
        self.scale = scale
        self.h = image_height
        self.w = image_width

        self.frame_num = 1

        preprocessing_vid = output_dir.joinpath("preprocessing.avi")
        contact_length_vid = output_dir.joinpath("contact-length-extraction.avi")
        inside_contour_vid = output_dir.joinpath("inside-contour-extraction.avi")
        self.thickness_graph_anim = output_dir.joinpath("chip-thickness-evolution.avi")

        codec = cv.VideoWriter_fourcc(*'mp4v')
        self.preprocessing_vid_writer = cv.VideoWriter(str(preprocessing_vid), codec, 30, (image_width, image_height))
        self.contact_vid_writer = cv.VideoWriter(str(contact_length_vid), codec, 30, (image_width, image_height))
        self.inside_vid_writer = cv.VideoWriter(str(inside_contour_vid), codec, 30, (image_width, image_height))
        self.thickness_animator = GraphAnimator(
            plot_configs=(
                {'linestyle': '-', 'marker': None, 'label': 'thickness'},
                {'linestyle': '-', 'marker': None, 'label': 'smoothed thickness'},
                {'linestyle': '-', 'marker': None, 'label': 'rough thickness'}
            ),
            scatter_configs=(
                {'s': 200, 'c': 'red', 'marker': 'o', 'label': 'rough peaks, used to compute the thickness quasiperiod'},
                {'s': 200, 'c': 'black', 'marker': 'v', 'label': 'valleys'},
                {'s': 200, 'c': 'black', 'marker': '^', 'label': 'peaks'}
            ),
            xlabel='inside contour point index',
            ylabel='chip thickness (µm)',
        )

    def render_frame(
        self,
        input_img: ColorImage,
        preproc_img: GrayImage,
        main_ft: MainFeatures,
        tip_ft: ToolTipFeatures,
        chip_ft: ChipFeatures,
        thk_an: ThicknessAnalysis
    ) -> None:
        self.preprocessing_vid_writer.write(cv.cvtColor(preproc_img, cv.COLOR_GRAY2BGR))

        contact_render = input_img.copy()
        render_contact_features(self.frame_num, contact_render, main_ft, tip_ft, chip_ft.contact_ft)
        self.contact_vid_writer.write(contact_render)

        inside_render = input_img.copy()
        render_inside_features(self.frame_num, inside_render, main_ft, tip_ft, chip_ft.inside_ft)
        self.inside_vid_writer.write(inside_render)

        thickness = self.scale * chip_ft.inside_ft.thickness
        smoothed = self.scale * thk_an.smoothed_thk
        rough = self.scale * thk_an.rough_thk
        rough_spikes = np.column_stack((thk_an.rough_spike_indices, rough[thk_an.rough_spike_indices]))
        valleys = np.column_stack((thk_an.valley_indices, smoothed[thk_an.valley_indices]))
        spikes = np.column_stack((thk_an.spike_indices, smoothed[thk_an.spike_indices]))
        self.thickness_animator.add_frame((thickness, smoothed, rough), (rough_spikes, valleys, spikes))

        self.frame_num += 1

    def no_render(self, input_img: ColorImage, preproc_img: GrayImage) -> None:
        self.preprocessing_vid_writer.write(cv.cvtColor(preproc_img, cv.COLOR_GRAY2BGR))
        self.contact_vid_writer.write(input_img)
        self.inside_vid_writer.write(input_img)
        self.thickness_animator.add_frame(
            (EMPTY_FLOAT_ARRAY, EMPTY_FLOAT_ARRAY, EMPTY_FLOAT_ARRAY),
            (EMPTY_FLOAT_PT_ARRAY, EMPTY_FLOAT_PT_ARRAY, EMPTY_FLOAT_PT_ARRAY)
        )
        self.frame_num += 1

    def release(self) -> None:
        self.preprocessing_vid_writer.release()
        self.contact_vid_writer.release()
        self.inside_vid_writer.release()
        self.thickness_animator.save(self.thickness_graph_anim)
