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

    def save_animation(animation_path: Path) -> None:
        ...


class AbstractAnalysisRenderer(abc.ABC):
    @abc.abstractmethod
    def render_frame(
        self,
        main_ft: MainFeatures,
        contact_ft: ContactFeatures,
        inside_ft: InsideFeatures,
        contact_len: float,
        thickness_analysis: ThicknessAnalysis,
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
        _contact_len: float,
        _thickness_analysis: ThicknessAnalysis,
    ) -> None:
        return

    def release(self) -> None:
        return


"""
TODO: AnalysisRenderer
- video which illustrates the contact length measurement
- video which illustrates the detection of the chip inside contour
- graph animation which shows the chip thickness measured on each input frame
- single graph shiwh shows the evolution of the contact length vs the frame
"""
class AnalysisRenderer(AbstractAnalysisRenderer):
    def __init__(self, render_dir: Path, scale: float, h: int, w: int) -> None:
        self.scale = scale

        contact_render_path = render_dir.joinpath("contact-length-extraction.avi")
        inside_render_path = render_dir.joinpath("inside-contour-extraction.avi")
        self.thickness_animation_path = render_dir.joinpath("chip-thickness-evolution.avi")

        codec = cv.VideoWriter_fourcc(*'mp4v')
        self.contact_vid_writer = cv.VideoWriter(str(contact_render_path), codec, 30, (h, w))
        self.inside_vid_writer = cv.VideoWriter(str(inside_render_path), codec, 30, (h, w))

        self.thickness_animator = GraphAnimator()  # MOCK: GraphAnimator

        self.contact_lengths: list[float] = []

    def render_frame(
        self,
        main_ft: MainFeatures,
        contact_ft: ContactFeatures,
        inside_ft: InsideFeatures,
        contact_len: float,
        thickness_analysis: ThicknessAnalysis,
    ) -> None:
        ...
        self.contact_lengths.append(self.scale * contact_len)


    def release(self) -> None:
        self.contact_vid_writer.release()
        self.inside_vid_writer.release()
        self.thickness_animator.save_animation(self.thickness_animation_path)



