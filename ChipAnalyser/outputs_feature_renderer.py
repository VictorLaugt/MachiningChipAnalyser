from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from type_hints import ColorImage, GrayImage
    from pathlib import Path
    from geometrical_analysis import GeometricalFeatures
    from features_thickness import ThicknessAnalysis

import abc
import cv2 as cv

from features_contact import render_contact_features
from features_thickness import render_inside_features


class AbstractFeatureRenderer(abc.ABC):
    @abc.abstractmethod
    def render_frame(
        self,
        input_img: ColorImage,
        preproc_img: GrayImage,
        geom_ft: GeometricalFeatures,
        thickness_analysis: ThicknessAnalysis
    ) -> None:
        pass

    @abc.abstractmethod
    def release(self) -> None:
        pass

    def __enter__(self) -> AbstractFeatureRenderer:
        return self

    def __exit__(self, _exc_type, _exc_value, _exc_traceback) -> None:
        self.release()


class NoRendering(AbstractFeatureRenderer):
    def render_frame(
        self,
        _input_img: ColorImage,
        _preproc_img: GrayImage,
        _geom_ft: GeometricalFeatures,
        _thickness_analysis: ThicknessAnalysis
    ) -> None:
        return

    def release(self) -> None:
        return


class FeatureRenderer(AbstractFeatureRenderer):
    def __init__(self, output_dir: Path, scale: float, image_height: int, image_width: int) -> None:
        self.scale = scale
        self.h = image_height
        self.w = image_width

        contact_length_vid = output_dir.joinpath("contact-length-extraction.avi")
        inside_contour_vid = output_dir.joinpath("inside-contour-extraction.avi")

        codec = cv.VideoWriter_fourcc(*'mp4v')
        self.contact_vid_writer = cv.VideoWriter(str(contact_length_vid), codec, 30, (image_width, image_height))
        self.inside_vid_writer = cv.VideoWriter(str(inside_contour_vid), codec, 30, (image_width, image_height))

    def render_frame(
        self,
        frame_num: int,
        input_img: ColorImage,
        _preproc_img: GrayImage,
        geom_ft: GeometricalFeatures,
    ) -> None:
        contact_render = input_img.copy()
        render_contact_features(frame_num, contact_render, geom_ft.main_ft, geom_ft.contact_ft)
        self.contact_vid_writer.write(contact_render)

        inside_render = input_img.copy()
        render_inside_features(frame_num, inside_render, geom_ft.main_ft, geom_ft.inside_ft)
        self.inside_vid_writer.write(inside_render)

    def release(self) -> None:
        self.contact_vid_writer.release()
        self.inside_vid_writer.release()
