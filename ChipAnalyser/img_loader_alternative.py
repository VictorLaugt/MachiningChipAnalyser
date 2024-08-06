from __future__ import annotations
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from typing import Iterator, Container, Sequence
    from type_hints import GrayImage

from pathlib import Path

import skimage.io
import skvideo.io

from img_loader import AbstractImageLoader, ImageLoadingError


class ImageDirectoryLoader(AbstractImageLoader):
    """Image loader for loading images from a directory."""

    def __init__(self, image_dir: Path, image_suffixes: Container[str], batch_size: int) -> None:
        img_paths = (file for file in image_dir.iterdir() if file.suffix in image_suffixes)
        self.img_paths: list[Path] = sorted(img_paths, key=(lambda file: file.name))
        if len(self.img_paths) == 0:
            raise ImageLoadingError
        super().__init__(batch_size)

    def img_shape(self) -> tuple[int, int]:
        return skimage.io.imread(str(self.img_paths[0])).shape[:2]

    def img_nb(self) -> int:
        return len(self.img_paths)

    def img_iter(self) -> Iterator[GrayImage]:
        for img_path in self.img_paths:
            yield skimage.io.imread(str(img_path), as_gray=True)

    def release(self) -> None:
        return


class VideoFrameLoader(AbstractImageLoader):
    """Image loader for loading frames from a video file."""

    def __init__(self, video_path: Path, batch_size: int) -> None:
        video_metadata = skvideo.io.ffprobe(video_path).get('video')
        if video_metadata is None:
            raise ImageLoadingError
        self.h = int(video_metadata['@height'])
        self.w = int(video_metadata['@width'])
        self.frame_nb = int(video_metadata['@nb_frames'])

        self.reader = skvideo.io.vreader(video_path)
        self.video_path = video_path
        super().__init__(batch_size)

    def img_shape(self) -> tuple[int, int]:
        return self.h, self.w

    def img_nb(self) -> int:
        return self.frame_nb

    def img_iter(self) -> Iterator[GrayImage]:
        # NOTE: requires numpy<1.24 where np.float is still an authorized alias to float
        for img in skvideo.io.vreader(str(self.video_path), as_grey=True):
            yield img.squeeze()

    def release(self) -> None:
        return
