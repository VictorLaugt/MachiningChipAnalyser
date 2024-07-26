from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    from pathlib import Path
    from type_hints import Image

import abc
import cv2 as cv


class AbstractVideoPlayer(abc.ABC):
    def __init__(self):
        self.last_info_length = 0

    def print_info(self, info_msg: str) -> None:
        pad = max(0, self.last_info_length - len(info_msg))
        print(info_msg + (pad * ' '), end='\r')
        self.last_info_length = len(info_msg)

    def erase_info(self) -> None:
        print(' ' * self.last_info_length, end='\r')
        self.last_info_length = 0

    def play(self) -> None:
        continue_playing = True
        while continue_playing:
            self.step()
            key = cv.waitKey(30) & 0xFF
            if key == 113:  # Q => quit
                continue_playing = False
            elif key == 32:  # Space => pause
                continue_playing = self.pause()
                self.erase_info()
        cv.destroyAllWindows()
        self.stop()

    def pause(self) -> None:
        while True:
            self.print_info(self.frame_count_message())
            key = cv.waitKey(30)
            if key == 32:  # Space => play
                return True
            elif key == 113:  # Q => quit
                return False
            elif key == 83 or key == 110 or key == 54:  # Right arrow or N or 6 => step
                self.step()
            elif key == 81 or key == 112 or key == 52:  # Left arrow or P or 4 => rewind
                self.rewind()

    @abc.abstractmethod
    def frame_count_message(self) -> str:
        pass

    @abc.abstractmethod
    def step(self) -> None:
        pass

    @abc.abstractmethod
    def rewind(self) -> None:
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        pass


class VideoFilePlayer(AbstractVideoPlayer):
    def __init__(self, video_file_path: Path, window_name: str):
        super().__init__()
        self.reader = cv.VideoCapture(str(video_file_path))
        self.window_name = window_name

    def frame_count_message(self) -> str:
        current_frame = self.reader.get(cv.CAP_PROP_POS_FRAMES)
        total_frame = self.reader.get(cv.CAP_PROP_FRAME_COUNT)
        return f"frame {current_frame}/{total_frame}"

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

    def stop(self) -> None:
        self.reader.release()


class VideoImgSeqPlayer(AbstractVideoPlayer):
    def __init__(self, image_sequence: Sequence[Image], window_name: str):
        super().__init__()
        self.image_sequence = image_sequence
        self.window_name = window_name
        self.frame_index = 0

    def frame_count_message(self) -> str:
        return f"frame {self.frame_index}/{len(self.image_sequence)}"

    def step(self) -> None:
        if self.frame_index >= len(self.image_sequence):
            self.frame_index = 0
        else:
            cv.imshow(self.window_name, self.image_sequence[self.frame_index])
            self.frame_index += 1

    def rewind(self) -> None:
        i = self.frame_index - 2
        i = len(self.image_sequence) - 1 if i < 0 else i
        self.frame_index = i
        self.step()

    def stop(self) -> None:
        return
