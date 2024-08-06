from __future__ import annotations
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from typing import Iterator, Container, Sequence
    from type_hints import GrayImage

import abc
from pathlib import Path

from skimage.util import img_as_ubyte
import skimage.io
import skvideo.io


class ImageLoadingError(Exception):
    """Exception raised for errors in the image loading process."""
    pass


class AbstractImageLoader(abc.ABC):
    """Abstract base class for iterating in batches on images of the same shape."""

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
    def img_shape(self) -> tuple[int, int]:
        """Return the shape of the images. It can be (h, w) for gray images or
        (h, w, 3) for color images.
        """
        pass

    @abc.abstractmethod
    def img_nb(self) -> int:
        """Return the number of images."""
        pass

    @abc.abstractmethod
    def img_iter(self) -> Iterator[GrayImage]:
        """Iterate over all the images, one by one."""
        pass


    def batch_nb(self) -> int:
        """Return the number of batches."""
        return len(self.batch_sizes)

    def img_batch_iter(self) -> Iterator[Sequence[GrayImage]]:
        """Iterate in batches over the images."""
        img_itr = self.img_iter()
        for batch_size in self.batch_sizes:
            yield [next(img_itr) for i in range(batch_size)]


    @abc.abstractmethod
    def release(self) -> None:
        """Release any resources held by the image loader."""
        pass

    def __enter__(self) -> AbstractImageLoader:
        return self

    def __exit__(self, _exc_type, _exc_value, _exc_traceback) -> None:
        self.release()


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
            yield img_as_ubyte(skimage.io.imread(str(img_path), as_gray=True))

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

        self.video_path = video_path
        super().__init__(batch_size)

    def img_shape(self) -> tuple[int, int]:
        return self.h, self.w

    def img_nb(self) -> int:
        return self.frame_nb

    def img_iter(self) -> Iterator[GrayImage]:
        # NOTE: requires numpy<1.24 so that np.float is still an authorized alias to float
        for img in skvideo.io.vreader(str(self.video_path), as_grey=True):
            yield img_as_ubyte(img.squeeze())

    def release(self) -> None:
        return
