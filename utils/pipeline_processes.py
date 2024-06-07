from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable, Sequence, Callable, TypeVar
    from numpy import ndarray
    Image = TypeVar("Image", bound=ndarray)

import numpy as np
import cv2 as cv

from pathlib import Path
from collections import OrderedDict

import video


def img_seq_equals(img_itr_1: Iterable[Image], img_itr_2: Iterable[Image]) -> bool:
    return all(np.array_equal(img_1, img_2) for img_1, img_2 in zip(img_itr_1, img_itr_2))


class PipelineError(Exception):
    pass

class PipelineNotFinishedError(PipelineError):
    def __init__(self):
        super().__init__("Pipeline execution not finished")

class InvalidStepNameError(PipelineError):
    pass

class AlreadyExistingStepNameError(InvalidStepNameError):
    def __init__(self, step_name: str):
        super().__init__(f"Step name '{step_name}' already exists")

class NotExistingStepNameError(InvalidStepNameError):
    def __init__(self, step_name: str):
        super().__init__(f"Step name '{step_name}' does not exist")

class InvalidResultDirError(PipelineError, FileExistsError):
    def __init__(self, result_dir: Path):
        super().__init__(f"Result directory '{result_dir}' is not a directory")


class Pipeline:
    _INPUT_STEP_NAME = "input"

    def __init__(self):
        self.steps = OrderedDict()
        self.operations = []

        self.input_image_sequence = None
        self.image_sequences = []
        self.result_dir = Path()
        self.finished = False

    def copy(self) -> Pipeline:
        copy = super().__new__(Pipeline)

        copy.steps = self.steps.copy()
        copy.operations = self.operations.copy()

        copy.input_image_sequence = self.input_image_sequence
        copy.image_sequences = []
        copy.result_dir = self.result_dir
        copy.finished = False

        return copy


    def __len__(self) -> int:
        return len(self.operations)

    def add(self, name: str, operation: Callable[[Image], Image]) -> None:
        if name in self.steps.keys():
            raise AlreadyExistingStepNameError(name)

        self.steps[name] = len(self)
        self.operations.append(operation)

    def then(self, other: Pipeline) -> Pipeline:
        pipe = self.copy()
        for name, operation in zip(other.steps.keys(), other.operations):
            pipe.add(name, operation)
        return pipe


    def run(self, input_image_sequence: Sequence[Image], result_dir_path: Path = None) -> None:
        if result_dir_path is not None:
            self.result_dir = result_dir_path

        self.finished = False
        self.input_image_sequence = input_image_sequence

        last_img_seq = self.input_image_sequence
        for step_name, step_id in self.steps.items():
            print(f"Running {step_name} ...")
            op = self.operations[step_id]
            last_img_seq = [op(img) for img in last_img_seq]
            self.image_sequences.append(last_img_seq)

        self.finished = True


    def _get(self, step_name: str) -> Sequence[Image]:
        if step_name == self._INPUT_STEP_NAME:
            return self.input_image_sequence
        step_id = self.steps.get(step_name)
        if step_id is None:
            raise NotExistingStepNameError(step_name)
        return self.image_sequences[self.steps[step_name]]

    def get(self, step_name: str) -> Sequence[Image]:
        if not self.finished:
            raise PipelineNotFinishedError()
        return self._get(step_name)

    def get_input(self) -> Sequence[Image]:
        if not self.finished:
            raise PipelineNotFinishedError()
        return self.input_image_sequence

    def get_output(self) -> Sequence[Image]:
        if not self.finished:
            raise PipelineNotFinishedError()
        if len(self) == 0:
            return self.input_image_sequence
        return self.image_sequences[-1]


    def show_frame(self, frame_index: int) -> None:
        if not self.finished:
            raise PipelineNotFinishedError()

        cv.imshow(self._INPUT_STEP_NAME, self.input_image_sequence[frame_index])
        for step_name, step_id in self.steps.items():
            cv.imshow(step_name, self.image_sequences[step_id][frame_index])

        while cv.waitKey(30) != 113:
            pass
        cv.destroyAllWindows()

    def show_video(self) -> None:
        if not self.finished:
            raise PipelineNotFinishedError()

        if not self.result_dir.exists():
            self.result_dir.mkdir()
        elif not self.result_dir.is_dir():
            raise InvalidResultDirError(self.result_dir)

        video.create_from_gray(
            self.input_image_sequence,
            str(self.result_dir.joinpath(f"{self._INPUT_STEP_NAME}.avi"))
        )
        for step_name, step_id in self.steps.items():
            video.create_from_gray(
                self.image_sequences[step_id],
                str(self.result_dir.joinpath(f"{step_name}.avi"))
            )

        video.play(self.result_dir.joinpath(f"{self._INPUT_STEP_NAME}.avi"))
        for step_name in self.steps.keys():
            video.play(self.result_dir.joinpath(f"{step_name}.avi"))


    def compare_frames(self, frame_index: int, step_names: Sequence[str]) -> None:
        if not self.finished:
            raise PipelineNotFinishedError()

        imgs = []
        for name in step_names:
            imgs.append(self._get(name)[frame_index])
        cv.imshow("_".join(step_names), np.vstack(imgs))

        while cv.waitKey(30) != 113:
            pass
        cv.destroyAllWindows()

    def compare_videos(self, step_names: Sequence[str]) -> None:
        if not self.finished:
            raise PipelineNotFinishedError()

        img_seqs = []
        for name in step_names:
            img_seqs.append(self._get(name))
        stacked_img_seq = []
        for frame_index in range(len(self.image_sequences[0])):
            stacked_img_seq.append(np.vstack([seq[frame_index] for seq in img_seqs]))

        comparison_video_path = self.result_dir.joinpath(f"{'_'.join(step_names)}.avi")
        video.create_from_gray(stacked_img_seq, str(comparison_video_path))
        video.play(comparison_video_path)
