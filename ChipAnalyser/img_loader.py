from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator, Container
    from type_hints import Image


import abc
import cv2 as cv
from pathlib import Path


class AbstractImageLoader(abc.ABC):
    @abc.abstractmethod
    def img_shape(self) -> tuple[int, int] | tuple[int, int, int]:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Image]:
        pass

    @abc.abstractmethod
    def release(self) -> None:
        pass

    def __enter__(self) -> AbstractImageLoader:
        return self

    def __exit__(self, _exc_type, _exc_value, _exc_traceback) -> None:
        self.release()


class ImageDirectoryLoader(AbstractImageLoader):
    def __init__(self, image_dir: Path, image_suffixes: Container[str]) -> None:
        img_paths = (file for file in image_dir.iterdir() if file.suffix in image_suffixes)
        self.img_paths: list[Path] = sorted(img_paths, key=(lambda file: file.name))

    def img_shape(self) -> tuple[int, int] | tuple[int, int, int]:
        return cv.imread(str(self.img_paths[0])).shape

    def __len__(self) -> int:
        return len(self.img_paths)

    def __iter__(self) -> Iterator[Image]:
        for path in self.img_paths:
            yield cv.imread(str(path))

    def release(self) -> None:
        return


# TODO: VideoFrameLoader
class VideoFrameLoader(AbstractImageLoader):
    def __init__(self, video_path: Path) -> None:
        self.reader = cv.VideoCapture(str(video_path))

    def img_shape(self) -> tuple[int, int, int]:
        return (
            int(self.reader.get(cv.CAP_PROP_FRAME_HEIGHT)),
            int(self.reader.get(cv.CAP_PROP_FRAME_WIDTH)),
            3
        )

    def __len__(self) -> int:
        return int(self.reader.get(cv.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[Image]:
        while True:
            ret, frame = self.reader.read()
            if not ret:
                break
            yield frame

    def release(self) -> None:
        self.reader.release()
