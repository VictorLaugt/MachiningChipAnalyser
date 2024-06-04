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

class InvalidResultDirError(PipelineError, FileExistsError):
    def __init__(self, result_dir: Path):
        super().__init__(f"Result directory '{result_dir}' is not a directory")


class Pipeline:
    def __init__(self):
        self.steps = OrderedDict()
        self.operations = []

        self.input_image_sequence = None
        self.image_sequences = []
        self.result_dir = Path()
        self.finished = False


    def __len__(self) -> int:
        return len(self.operations)

    def add(self, name: str, operation: Callable[[Image], Image]) -> None:
        if name in self.steps.keys():
            raise InvalidStepNameError(f"Step name '{name}' already exists")

        self.steps[name] = len(self)
        self.operations.append(operation)


    def run(self, input_image_sequence: Sequence[Image], result_dir_path: Path = None) -> None:
        if result_dir_path is not None:
            self.result_dir = result_dir_path

        self.finished = False
        self.input_image_sequence = input_image_sequence

        running_img_seq = self.input_image_sequence
        for step_name, step_id in self.steps.items():
            print(f"Running {step_name} ...")
            op = self.operations[step_id]
            running_img_seq = [op(img) for img in running_img_seq]
            self.image_sequences.append(running_img_seq)

        self.finished = True


    def get(self, name: str) -> Sequence[Image]:
        if not self.finished:
            raise PipelineNotFinishedError()
        return self.image_sequences[self.steps[name]]

    def get_input(self) -> Sequence[Image]:
        if not self.finished:
            raise PipelineNotFinishedError()
        return self.input_image_sequence

    def get_output(self) -> Sequence[Image]:
        if not self.finished:
            raise PipelineNotFinishedError()
        return self.image_sequences[-1]


    def show_samples(self, idx: int) -> None:
        if not self.finished:
            raise PipelineNotFinishedError()

        cv.imshow("original", self.input_image_sequence[idx])
        for step_name, step_id in self.steps.items():
            cv.imshow(step_name, self.image_sequences[step_id][idx])
        while cv.waitKey(30) != 113:
            pass
        cv.destroyAllWindows()

    def show_videos(self) -> None:
        if not self.finished:
            raise PipelineNotFinishedError()

        if not self.result_dir.exists():
            self.result_dir.mkdir()
        elif not self.result_dir.is_dir():
            raise InvalidResultDirError(self.result_dir)

        video.create_from_gray(self.input_image_sequence, str(self.result_dir.joinpath("original.avi")))
        for step_name, step_id in self.steps.items():
            video.create_from_gray(self.image_sequences[step_id], str(self.result_dir.joinpath(f"{step_name}.avi")))

        video.play(self.result_dir.joinpath("original.avi"))
        for step_name in self.steps.keys():
            video.play(self.result_dir.joinpath(f"{step_name}.avi"))
