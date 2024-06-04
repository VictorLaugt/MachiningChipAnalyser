"""
Normalization
Threshold binarization
Morphological edge extraction (subtraction of eroded image from binary image)
"""

import cv2 as cv

import utils


pipeline = utils.Pipeline()

pipeline.add("norm", lambda img: cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
pipeline.add("binary", lambda img: cv.threshold(img, 15, 255, cv.THRESH_BINARY)[1])
pipeline.add("edge", lambda img: cv.morphologyEx(img, cv.MORPH_GRADIENT, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))))


if __name__ == '__main__':
    from pathlib import Path
    import image_loader

    dir_path = Path("imgs", "vertical")
    # dir_path = Path("imgs", "diagonal")
    loader = image_loader.ImageLoaderColorConverter(dir_path, cv.COLOR_RGB2GRAY)

    pipeline.run(loader)
    pipeline.show_samples(20)
    pipeline.show_videos()
