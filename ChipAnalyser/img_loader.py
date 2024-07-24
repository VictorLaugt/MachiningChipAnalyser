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


# TODO: VideoFrameLoader
class VideoFrameLoader(AbstractImageLoader):
    def __init__(self, video_path: Path) -> None:
        ...

    def img_shape(self) -> tuple[int, int] | tuple[int, int, int]:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[Image]:
        ...
