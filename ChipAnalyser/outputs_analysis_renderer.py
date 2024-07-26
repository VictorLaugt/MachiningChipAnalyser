from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from type_hints import ColorImage, GrayImage
    from pathlib import Path
    from analysis import GeometricalFeatures
    from features_main import MainFeatures
    from features_contact import ContactFeatures
    from features_thickness import InsideFeatures, ThicknessAnalysis

import abc
import numpy as np
import cv2 as cv

from features_contact import render_contact_features


# TODO: GraphAnimator
class GraphAnimator:
    ...

    def save_animation(self, animation_path: Path) -> None:
        ...


class AbstractAnalysisRenderer(abc.ABC):
    @abc.abstractmethod
    def render_frame(
        self,
        input_img: ColorImage,
        preprocessed_img: GrayImage,
        geom_ft: GeometricalFeatures,
        contact_len: float,
        thickness_analysis: ThicknessAnalysis
    ) -> None:
        pass

    @abc.abstractmethod
    def release(self) -> None:
        pass


class NoRendering(AbstractAnalysisRenderer):
    def render_frame(self, *_unused_args) -> None:
        return

    def release(self) -> None:
        return


"""
TODO: AnalysisRenderer
- video which illustrates the contact length measurement
- video which illustrates the detection of the chip inside contour
- graph animation which shows the chip thickness measured on each input frame
- single graph which shows the evolution of the contact length vs the frame
"""
class AnalysisRenderer(AbstractAnalysisRenderer):
    def __init__(
        self,
        scale: float,
        image_height: int,
        image_width: int,
        contact_length_vid: Path,
        inside_contour_vid: Path,
        thickness_graph: Path,
        contact_length_graph: Path
    ) -> None:
        self.h = image_height
        self.w = image_width
        self.scale = scale

        contact_render_path = render_dir.joinpath("contact-length-extraction.avi")
        inside_render_path = render_dir.joinpath("inside-contour-extraction.avi")
        self.thickness_animation_path = render_dir.joinpath("chip-thickness-evolution.avi")

        codec = cv.VideoWriter_fourcc(*'mp4v')
        self.contact_vid_writer = cv.VideoWriter(str(contact_render_path), codec, 30, (image_width, image_height))
        self.inside_vid_writer = cv.VideoWriter(str(inside_render_path), codec, 30, (image_width, image_height))

        self.thickness_animator = GraphAnimator()  # MOCK: GraphAnimator

        self.contact_lengths: list[float] = []

    def render_frame(
        self,
        input_img: ColorImage,
        preprocessed_img: GrayImage,
        geom_ft: GeometricalFeatures,
        contact_len: float,
        thickness_analysis: ThicknessAnalysis
    ) -> None:
        contact_render = input_img.copy()
        render_contact_features(contact_render, geom_ft.main_ft, geom_ft.contact_ft)
        self.contact_vid_writer.write(contact_render)

        self.contact_lengths.append(self.scale * contact_len)


    def release(self) -> None:
        self.contact_vid_writer.release()
        self.inside_vid_writer.release()
        self.thickness_animator.save_animation(self.thickness_animation_path)



