import abc
import cv2 as cv
from pathlib import Path


class AbstractImageLoader(abc.ABC):
    def __init__(self, image_dir: Path):
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        self.img_paths = sorted(
            (file for file in image_dir.iterdir() if file.suffix == '.bmp'),
            key=lambda file: file.name
        )

    def __len__(self):
        return len(self.img_paths)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @abc.abstractmethod
    def __getitem__(self, index):
        pass


class ImageLoader(AbstractImageLoader):
    def __getitem__(self, index):
        return cv.imread(str(self.img_paths[index]))


class ImageLoaderColorConverter(AbstractImageLoader):
    def __init__(self, image_dir: Path, convert_code: int):
        super().__init__(image_dir)
        self.convert_code = convert_code

    def __getitem__(self, index):
        return cv.cvtColor(cv.imread(str(self.img_paths[index])), self.convert_code)
