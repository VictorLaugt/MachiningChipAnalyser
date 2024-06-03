import numpy as np
import cv2 as cv

from collections import OrderedDict

import video


def img_seq_equals(img_seq_1, img_seq_2):
    return all(np.array_equal(img_1, img_2) for img_1, img_2 in zip(img_seq_1, img_seq_2))


class PipelineError(Exception):
    pass

class PipelineNotFinishedError(PipelineError):
    def __init__(self):
        super().__init__("Pipeline execution not finished")

class InvalidStepNameError(PipelineError):
    pass


class Pipeline:
    def __init__(self):
        self.steps = OrderedDict()
        self.operations = []

        self.input_image_sequence = None
        self.image_sequences = []
        self.finished = False


    def __len__(self):
        return len(self.operations)

    def add(self, name, operation):
        if name in self.steps.keys():
            raise InvalidStepNameError(f"Step name '{name}' already exists")

        self.steps[name] = len(self)
        self.operations.append(operation)


    def run(self, input_image_sequence):
        self.finished = False
        self.input_image_sequence = input_image_sequence

        running_img_seq = self.input_image_sequence
        for step_name, step_id in self.steps.items():
            print(f"Running {step_name} ...")
            op = self.operations[step_id]
            running_img_seq = [op(img) for img in running_img_seq]
            self.image_sequences.append(running_img_seq)

        self.finished = True


    def get(self, name):
        if not self.finished:
            raise PipelineNotFinishedError()
        return self.image_sequences[self.steps[name]]

    def get_input(self):
        if not self.finished:
            raise PipelineNotFinishedError()
        return self.input_image_sequence

    def get_output(self):
        if not self.finished:
            raise PipelineNotFinishedError()
        return self.image_sequences[-1]


    def show_samples(self, idx):
        cv.imshow("original", self.input_image_sequence[idx])
        for step_name, step_id in self.steps.items():
            cv.imshow(step_name, self.image_sequences[step_id][idx])
        while cv.waitKey(30) != 113:
            pass
        cv.destroyAllWindows()

    def show_videos(self):
        video.create_from_gray(self.input_image_sequence, "original.avi")
        for step_name, step_id in self.steps.items():
            video.create_from_gray(self.image_sequences[step_id], f"{step_name}.avi")

        video.play("original.avi")
        for step_name in self.steps.keys():
            video.play(f"{step_name}.avi")
