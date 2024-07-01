from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable, Sequence, Callable, TypeVar
    from numpy import ndarray
    Image = TypeVar("Image", bound=ndarray)
    Feature = TypeVar("Feature")
    Operation = Callable[..., Image]
    FeatureExtractor = Callable[..., Feature]

import numpy as np
import cv2 as cv

from pathlib import Path
from collections import OrderedDict

import video


def img_seq_equals(img_itr_1: Iterable[Image], img_itr_2: Iterable[Image]) -> bool:
    return all(np.array_equal(img_1, img_2) for img_1, img_2 in zip(img_itr_1, img_itr_2))


def rgb_image(img: Image) -> Image:
    if img.ndim == 2:    # gray (n, h, w)
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    elif img.ndim == 3:  # rgb  (n, h, w, 3)
        return img
    elif img.ndim == 4:  # rgba (n, h, w, 4)
        return cv.cvtColor(img, cv.COLOR_RGBA2RGB)
    else:
        raise ValueError(f"Unknown array shape for an image: {img.shape}")


class DagError(Exception):
    pass

class DagNotFinishedError(DagError):
    def __init__(self):
        super().__init__("processing execution not finished")

class InvalidStepNameError(DagError):
    pass

class AlreadyExistingStepNameError(InvalidStepNameError):
    def __init__(self, step_name: str):
        super().__init__(f"Step name '{step_name}' already exists")

class NotExistingStepNameError(InvalidStepNameError):
    def __init__(self, step_name: str):
        super().__init__(f"Step name '{step_name}' does not exist")

class InvalidOutputDirError(DagError, FileExistsError):
    def __init__(self, output_dir: Path):
        super().__init__(f"Output directory '{output_dir}' is not a directory")


class DagProcessNode:
    def __init__(self, operation: Callable, node_input_ids: Sequence[int]):
        self.operation = operation
        self.node_input_ids = node_input_ids


# len(self.node_lists) == n
# len(self.steps) == len(self.image_sequences) == self.next_id == n + 1
class DagProcessVizualiser:
    def __init__(self):
        self.next_id = 1
        self.steps = OrderedDict((("input", 0),))
        self.node_list: list[DagProcessNode] = []

        self.image_sequences: list[Sequence[Image]] = []
        self.feature_extractions: list[Feature] = []
        self.finished = False

    def copy(self) -> DagProcessVizualiser:
        copy = super().__new__(DagProcessVizualiser)
        copy.next_id = self.next_id
        copy.steps = self.steps.copy()
        copy.node_list = self.node_list.copy()

        copy.image_sequences = []
        copy.finished = False

        return copy

    def __repr__(self) -> str:
        return (
            f"DagProcess(\n"
            f"\tnext_id = {self.next_id}\n"
            f"\tsteps = {self.steps}\n"
            f"\tlen(node_list) = {len(self.node_list)}\n"
            f"\tlen(image_sequences) = {len(self.image_sequences)}\n"
            ")"
        )

    def _ensure_output_directory(self, output_dir: Path) -> None:
        if not output_dir.is_dir():
            try:
                output_dir.mkdir()
            except OSError:
                raise InvalidOutputDirError(output_dir)

    def _ensure_finished(self) -> None:
        if not self.finished:
            raise DagNotFinishedError()

    def _get_new_id(self) -> int:
        new_id = self.next_id
        self.next_id += 1
        return new_id

    def add(
            self,
            name: str,
            operation: Operation,
            node_input_steps: Iterable[str] = None
    ) -> None:
        if name in self.steps.keys():
            raise AlreadyExistingStepNameError(name)

        step_id = self._get_new_id()
        if node_input_steps is None:
            node_input_ids = [step_id-1]
        else:
            node_input_ids = [self.steps[step_name] for step_name in node_input_steps]
        node = DagProcessNode(operation, node_input_ids)

        self.steps[name] = step_id
        self.node_list.append(node)

    def run(self, input_image_sequence: Sequence[Image]) -> None:
        self.finished = False

        image_sequences = [None] * self.next_id
        image_sequences[0] = input_image_sequence

        step_iterator = iter(self.steps.items())
        next(step_iterator)
        for step_name, step_id in step_iterator:
            print(f"Running {step_name} ...")
            node = self.node_list[step_id - 1]
            node_inputs = [image_sequences[i] for i in node.node_input_ids]
            node_operation = node.operation
            image_sequences[step_id] = [node_operation(*imgs) for imgs in zip(*node_inputs)]

        self.image_sequences = image_sequences
        self.finished = True


    def _get(self, step_name: str) -> Sequence[Image]:
        step_id = self.steps.get(step_name)
        if step_id is None:
            raise NotExistingStepNameError(step_name)
        return self.image_sequences[self.steps[step_name]]

    def get(self, step_name: str) -> Sequence[Image]:
        self._ensure_finished()
        return self._get(step_name)

    def get_input(self) -> Sequence[Image]:
        self._ensure_finished()
        return self.image_sequences[0]

    def get_output(self) -> Sequence[Image]:
        self._ensure_finished()
        return self.image_sequences[-1]


    def _create_frame_comp(self, frames: Sequence[Image], horizontal: bool) -> Image:
        if any(f.ndim > 2 for f in frames):
            frames = [rgb_image(f) for f in frames]
        return np.hstack(frames) if horizontal else np.vstack(frames)

    def _create_video_comp(self, img_seqs: Iterable[Sequence[Image]], horizontal: bool, force_rgb: bool) -> Sequence[Image]:
        stack_function = np.hstack if horizontal else np.vstack
        stacked_img_seq = []
        for frames in zip(*img_seqs):
            if force_rgb or any(img.ndim > 2 for img in frames):
                frames = [rgb_image(img) for img in frames]
            stacked_img_seq.append(stack_function(frames))
        return stacked_img_seq

    def show_frames(self, frame_index: int) -> None:
        self._ensure_finished()
        for step_name, step_id in self.steps.items():
            cv.imshow(step_name, self.image_sequences[step_id][frame_index])
        while cv.waitKey(30) != 113:
            pass
        cv.destroyAllWindows()


    def show_frame_comp(self, frame_index: int, step_names: Sequence[str], horizontal: bool=False) -> None:
        self._ensure_finished()
        frames = [self._get(name)[frame_index] for name in step_names]
        stacked_img = self._create_frame_comp(frames, horizontal)
        cv.imshow("_".join(step_names), stacked_img)
        while cv.waitKey(30) != 113:
            pass
        cv.destroyAllWindows()

    def save_frame_comp(self, image_output_dir: Path, frame_index: int, step_names: Sequence[str], horizontal: bool=False) -> None:
        self._ensure_finished()
        self._ensure_output_directory(image_output_dir)
        frames = [self._get(name)[frame_index] for name in step_names]
        stacked_img = self._create_frame_comp(frames, horizontal)
        comparison_image_path = image_output_dir.joinpath(f"{'_'.join(step_names)}.png")
        cv.imwrite(str(comparison_image_path), stacked_img)


    def show_videos(self) -> None:
        self._ensure_finished()
        for step_name, step_id in self.steps.items():
            video.VideoImgSeqPlayer(self.image_sequences[step_id], step_name).play()

    def save_videos(self, video_output_dir: Path) -> None:
        self._ensure_finished()
        self._ensure_output_directory(video_output_dir)
        for step_name, step_id in self.steps.items():
            video.create_from_rgb(
                (rgb_image(img) for img in self.image_sequences[step_id]),
                video_output_dir.joinpath(f"{step_name}.avi")
            )


    def show_video_comp(self, step_names: Sequence[str], horizontal: bool=False) -> None:
        self._ensure_finished()
        img_seqs = [self._get(name) for name in step_names]
        stacked_img_seq = self._create_video_comp(img_seqs, horizontal, force_rgb=False)
        video.VideoImgSeqPlayer(stacked_img_seq, "_".join(step_names)).play()

    def save_video_comp(self, video_output_dir: Path, step_names: Sequence[str], horizontal: bool=False) -> None:
        self._ensure_finished()
        self._ensure_output_directory(video_output_dir)
        img_seqs = [self._get(name) for name in step_names]
        stacked_img_seq = self._create_video_comp(img_seqs, horizontal, force_rgb=True)
        comparison_video_path = video_output_dir.joinpath(f"{'_'.join(step_names)}.avi")
        video.create_from_rgb(stacked_img_seq, comparison_video_path)

