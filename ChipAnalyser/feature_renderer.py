from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
    from chip_extract import MainFeatures
    from contact_measurement import ContactFeatures
    from thickness_measurement import InsideFeatures

from contact_measurement import render_contact_features

import abc

import cv2 as cv


# TODO: GraphAnimator
class GraphAnimator:
    ...


class AbstractFeatureRenderer(abc.ABC):
    @abc.abstractmethod
    def render_frame(self, main_ft: MainFeatures, contact_ft: ContactFeatures, inside_ft: InsideFeatures) -> None:
        pass

    @abc.abstractmethod
    def release(self) -> None:
        pass


class NoRendering(AbstractFeatureRenderer):
    def render_frame(self, _main_ft: MainFeatures, _contact_ft: ContactFeatures, _inside_ft: InsideFeatures) -> None:
        return

    def release(self) -> None:
        return


# TODO: FeatureRenderer
class FeatureRenderer(AbstractFeatureRenderer):
    def __init__(self, render_dir: Path, h: int, w: int) -> None:
        codec = cv.VideoWriter_fourcc(*'mp4v')

        contact_render_path = str(render_dir.joinpath("contact.avi"))
        inside_contour_render_path = str(render_dir.joinpath("inside_contour.avi"))

        self.contact_vid_writer = cv.VideoWriter(contact_render_path, codec, 30, (h, w))
        self.inside_contour_vid_writer = cv.VideoWriter(inside_contour_render_path, codec, 30, (h, w))
        # self.thickness_animator = GraphAnimator()


    def render_frame(self, main_ft: MainFeatures, contact_ft: ContactFeatures, inside_ft: InsideFeatures) -> None:
        ...

    def release(self) -> None:
        ...
