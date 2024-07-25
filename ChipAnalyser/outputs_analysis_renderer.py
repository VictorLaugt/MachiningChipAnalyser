from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
    from features_main import MainFeatures
    from features_contact import ContactFeatures
    from features_thickness import InsideFeatures, ThicknessAnalysis

import abc
import cv2 as cv

from features_contact import render_contact_features


# TODO: GraphAnimator
class GraphAnimator:
    ...


class AbstractAnalysisRenderer(abc.ABC):
    @abc.abstractmethod
    def render_frame(
        self,
        main_ft: MainFeatures,
        contact_ft: ContactFeatures,
        inside_ft: InsideFeatures,
        thickness_analysis: ThicknessAnalysis,
        scale: float
    ) -> None:
        pass

    @abc.abstractmethod
    def release(self) -> None:
        pass


class NoRendering(AbstractAnalysisRenderer):
    def render_frame(
        self,
        _main_ft: MainFeatures,
        _contact_ft: ContactFeatures,
        _inside_ft: InsideFeatures,
        _thickness_analysis: ThicknessAnalysis,
        _scale: float
    ) -> None:
        return

    def release(self) -> None:
        return


# TODO: FeatureRenderer
class AnalysisRenderer(AbstractAnalysisRenderer):
    def __init__(self, render_dir: Path, h: int, w: int) -> None:
        codec = cv.VideoWriter_fourcc(*'mp4v')

        contact_render_path = str(render_dir.joinpath("contact.avi"))
        inside_contour_render_path = str(render_dir.joinpath("inside_contour.avi"))

        self.contact_vid_writer = cv.VideoWriter(contact_render_path, codec, 30, (h, w))
        self.inside_contour_vid_writer = cv.VideoWriter(inside_contour_render_path, codec, 30, (h, w))
        # self.thickness_animator = GraphAnimator()


    def render_frame(
        self,
        main_ft: MainFeatures,
        contact_ft: ContactFeatures,
        inside_ft: InsideFeatures,
        thickness_analysis: ThicknessAnalysis,
        scale: float
    ) -> None:
        ...

    def release(self) -> None:
        ...
