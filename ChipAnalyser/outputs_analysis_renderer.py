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
import cv2 as cv
import matplotlib.pyplot as plt

from features_contact import render_contact_features


# TODO: GraphAnimator
class GraphAnimator:
    ...

    def create_animation(self) -> go.Figure:
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

    @abc.abstractmethod
    def display_render(self) -> None:
        pass

class NoRendering(AbstractAnalysisRenderer):
    def render_frame(self, *_unused_args) -> None:
        return

    def release(self) -> None:
        return

    def display_render(self) -> None:
        return



"""
TODO: AnalysisRenderer
- video which illustrates the contact length measurement
- video which illustrates the detection of the chip inside contour
- graph animation which shows the chip thickness measured on each input frame
- single graph which shows the evolution of the contact length vs the frame
"""
class AnalysisRenderer(AbstractAnalysisRenderer):
    def __init__(self, output_dir: Path, scale: float, image_height: int, image_width: int) -> None:
        self.scale = scale
        self.h = image_height
        self.w = image_width

        contact_length_vid = output_dir.joinpath("contact-length-extraction.avi")
        inside_contour_vid = output_dir.joinpath("inside-contour-extraction.avi")
        self.thickness_graph_anim = output_dir.joinpath("chip-thickness-evolution.avi")

        codec = cv.VideoWriter_fourcc(*'mp4v')
        self.contact_vid_writer = cv.VideoWriter(str(contact_length_vid), codec, 30, (image_width, image_height))
        self.inside_vid_writer = cv.VideoWriter(str(inside_contour_vid), codec, 30, (image_width, image_height))
        self.thickness_animator = GraphAnimator()  # MOCK: GraphAnimator
        self.contact_lengths: list[float] = []

    def render_frame(
        self,
        input_img: ColorImage,
        preproc_img: GrayImage,
        geom_ft: GeometricalFeatures,
        contact_len: float,
        thickness_analysis: ThicknessAnalysis
    ) -> None:
        contact_render = input_img.copy()
        render_contact_features(contact_render, geom_ft.main_ft, geom_ft.contact_ft)
        self.contact_vid_writer.write(contact_render)

        self.contact_lengths.append(self.scale * contact_len)

    def create_contact_length_graph(self, display: bool) -> None:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.set_title("Contact length evolution")
        ax.set_ylabel("contact length (µm)")
        ax.set_xlabel("frame")
        ax.grid()
        ax.plot(range(1, len(self.contact_lengths)+1), self.contact_lengths)
        if display:
            plt.show()
        else:
            fig.savefig(self.contact_length_graph_path)
        plt.close('all')

    def release(self) -> None:
        self.contact_vid_writer.release()
        self.inside_vid_writer.release()

        self.create_contact_length_graph(display=False)
        self.thickness_animator.save_animation()

        # thickness_animation = self.thickness_animator.create_animation()
        # contact_length_graph = self.create_contact_length_graph()

        # self.thickness_animator.save_animation(self.thickness_animation_path)
        # contact_length_graph.write_image(self.contact_length_graph_path)

    def display_render(self) -> None:
        self.create_contact_length_graph(display=True)

