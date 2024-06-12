from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
    from typing import Iterable, TypeVar
    from numpy import ndarray
    GrayImg = TypeVar("GrayImg", bound=ndarray)
    RGBImg = TypeVar("RGBImg", bound=ndarray)

import cv2 as cv

def create_from_rgb(rgb_image_itr: Iterable[RGBImg], video_file_path: Path) -> None:
    rgb_image_itr = iter(rgb_image_itr)
    img = next(rgb_image_itr)

    codec = cv.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv.VideoWriter(str(video_file_path), codec, 30, (img.shape[1], img.shape[0]))

    for img in rgb_image_itr:
        vid_writer.write(img)

    vid_writer.release()


def create_from_gray(gray_image_itr: Iterable[GrayImg], video_file_path: Path) -> None:
    return create_from_rgb(
        map(lambda img: cv.cvtColor(img, cv.COLOR_GRAY2BGR), gray_image_itr),
        str(video_file_path)
    )


class _VideoPlayer:
    def __init__(self, reader: cv.VideoCapture, window_name: str):
        self.reader = reader
        self.window_name = window_name
        self.last_message_length = 0

    def print_frame_count(self) -> None:
        current_frame = self.reader.get(cv.CAP_PROP_POS_FRAMES)
        total_frame = self.reader.get(cv.CAP_PROP_FRAME_COUNT)
        message = f"frame {current_frame}/{total_frame}"
        pad = max(0, self.last_message_length - len(message))
        print(message + (pad * ' '), end='\r')
        self.last_message_length = len(message)

    def erase_frame_count(self) -> None:
        print(' ' * self.last_message_length, end='\r')
        self.last_message_length = 0

    def step(self) -> None:
        ret, frame = self.reader.read()
        if not ret:
            self.reader.set(cv.CAP_PROP_POS_FRAMES, 0)
        else:
            cv.imshow(self.window_name, frame)

    def rewind(self) -> None:
        i = self.reader.get(cv.CAP_PROP_POS_FRAMES) - 2
        i = self.reader.get(cv.CAP_PROP_FRAME_COUNT) - 1 if i < 0 else i
        self.reader.set(cv.CAP_PROP_POS_FRAMES, i)
        self.step()

    def pause(self) -> None:
        while True:
            self.print_frame_count()
            key = cv.waitKey(30)
            if key == 32:  # Space => play
                return True
            elif key == 113:  # Q => quit
                return False
            elif key == 83 or key == 110:  # Right arrow or N => step
                self.step()
            elif key == 81 or key == 112:  # Left arrow or P => rewind
                self.rewind()

    def play(self) -> None:
        continue_playing = True
        while continue_playing:
            self.step()
            key = cv.waitKey(30) & 0xFF
            if key == 113:  # Q => quit
                continue_playing = False
            elif key == 32:  # Space => pause
                continue_playing = self.pause()
                self.erase_frame_count()


def play(video_file_path: Path) -> None:
    video_reader = cv.VideoCapture(str(video_file_path))
    _VideoPlayer(video_reader, video_file_path.stem).play()
    video_reader.release()
    cv.destroyAllWindows()
