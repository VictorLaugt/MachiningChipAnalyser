from __future__ import annotations
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from typing import Iterator, Container, Sequence
    from type_hints import Image


import abc
import cv2 as cv
from pathlib import Path


class AbstractImageLoader(abc.ABC):
    def __init__(self, batch_size: int) -> None:
        img_nb = self.img_nb()
        div, mod = divmod(img_nb, batch_size)
        if mod == 0:
            self.batch_sizes = [batch_size] * div
        elif div == 0:
            self.batch_sizes = [img_nb]
        else:
            self.batch_sizes = [batch_size] * (div-1)
            self.batch_sizes.append(batch_size + mod)


    @abc.abstractmethod
    def img_shape(self) -> tuple[int, int] | tuple[int, int, int]:
        pass

    @abc.abstractmethod
    def img_nb(self) -> int:
        pass

    @abc.abstractmethod
    def img_iter(self) -> Iterator[Image]:
        pass


    def batch_nb(self) -> int:
        return len(self.batch_sizes)

    def img_batch_iter(self) -> Iterator[Sequence[Image]]:
        img_itr = self.img_iter()
        for batch_size in self.batch_sizes:
            yield [next(img_itr) for i in range(batch_size)]


    @abc.abstractmethod
    def release(self) -> None:
        pass

    def __enter__(self) -> AbstractImageLoader:
        return self

    def __exit__(self, _exc_type, _exc_value, _exc_traceback) -> None:
        self.release()


class ImageDirectoryLoader(AbstractImageLoader):
    def __init__(self, image_dir: Path, image_suffixes: Container[str], batch_size: int) -> None:
        img_paths = (file for file in image_dir.iterdir() if file.suffix in image_suffixes)
        self.img_paths: list[Path] = sorted(img_paths, key=(lambda file: file.name))
        super().__init__(batch_size)

    def img_shape(self) -> tuple[int, int] | tuple[int, int, int]:
        return cv.imread(str(self.img_paths[0])).shape

    def img_nb(self) -> int:
        return len(self.img_paths)

    def img_iter(self) -> Iterator[Image]:
        return (cv.imread(str(img_path)) for img_path in self.img_paths)

    def release(self) -> None:
        return


class VideoFrameLoader(AbstractImageLoader):
    def __init__(self, video_path: Path, batch_size: int) -> None:
        self.reader = cv.VideoCapture(str(video_path))
        super().__init__(batch_size)

    def img_shape(self) -> tuple[int, int, int]:
        return (
            int(self.reader.get(cv.CAP_PROP_FRAME_HEIGHT)),
            int(self.reader.get(cv.CAP_PROP_FRAME_WIDTH)),
            3
        )

    def img_nb(self) -> int:
        return int(self.reader.get(cv.CAP_PROP_FRAME_COUNT))

    def img_iter(self) -> Iterator[Image]:
        while True:
            ret, frame = self.reader.read()
            if not ret:
                break
            yield frame

    def release(self) -> None:
        self.reader.release()
