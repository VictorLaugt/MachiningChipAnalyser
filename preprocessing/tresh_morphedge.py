"""
Threshold binarization
Morphological edge extraction (subtraction of eroded image from binary image)
"""

import cv2 as cv

import utils


pipeline = utils.Pipeline()

pipeline.add("binary", lambda img: cv.threshold(img, 45, 255, cv.THRESH_BINARY)[1])
# pipeline.add("binary", lambda img: cv.threshold(img, 50, 255, cv.THRESH_BINARY)[1])
pipeline.add("edge", lambda img: cv.morphologyEx(img, cv.MORPH_GRADIENT, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))))


if __name__ == '__main__':
    from pathlib import Path
    import image_loader

    dir_path = Path("imgs", "vertical")
    # dir_path = Path("imgs", "diagonal")
    loader = image_loader.ImageLoaderColorConverter(dir_path, cv.COLOR_RGB2GRAY)

    pipeline.run(loader)
    pipeline.show_samples(40)
    pipeline.show_videos()
