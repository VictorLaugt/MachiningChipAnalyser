from __future__ import annotations
from typing import TYPE_CHECKING

from numpy import ndarray
if TYPE_CHECKING:
    import numpy as np
    from typing import Iterator

import abc
import cv2 as cv
from pathlib import Path


class AbstractImageLoader(abc.ABC):
    def __init__(self, image_dir: Path) -> None:
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        self.img_paths: list[np.ndarray] = sorted(
            (file for file in image_dir.iterdir() if file.suffix == '.bmp'),
            key=lambda file: file.name
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self._get_by_index(i)

    def __getitem__(self, x: int|slice) -> np.ndarray|list[np.ndarray]:
        if isinstance(x, int):
            return self._get_by_index(x)
        elif isinstance(x, slice):
            return self._get_by_slice(x)
        else:
            raise TypeError

    @abc.abstractmethod
    def _get_by_index(self, i: int) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _get_by_slice(self, s: slice) -> list[np.ndarray]:
        pass


class ImageLoader(AbstractImageLoader):
    def _get_by_index(self, i: int) -> np.ndarray:
        return cv.imread(str(self.img_paths[i]))

    def _get_by_slice(self, s: slice) -> list[np.ndarray]:
        return [cv.imread(str(path)) for path in self.img_paths[s]]


class ImageLoaderColorConverter(AbstractImageLoader):
    def __init__(self, image_dir: Path, convert_code: int) -> None:
        super().__init__(image_dir)
        self.convert_code = convert_code

    def _get_by_index(self, i: int) -> ndarray:
        return cv.cvtColor(cv.imread(str(self.img_paths[i])), self.convert_code)

    def _get_by_slice(self, s: slice) -> list[ndarray]:
        return [cv.cvtColor(cv.imread(str(path))) for path in self.img_paths[s]]
