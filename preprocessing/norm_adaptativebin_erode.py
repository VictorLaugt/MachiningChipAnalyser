"""
Gaussian blur
Normalization
Binarisation using an adaptative threshold
Moprhological erosion
"""

import cv2 as cv

import utils


processing = utils.DagProcess()

processing.add("blur", lambda img: cv.GaussianBlur(img, (9, 9), 0))
processing.add("norm", lambda img: cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
# processing.add("binary", lambda img: cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2))
processing.add("binary", lambda img: cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2))
processing.add("erode", lambda img: cv.erode(img, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))))
# processing.add("erode", lambda img: cv.morphologyEx(img, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)), iterations=2))


if __name__ == '__main__':
    from pathlib import Path
    import image_loader

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "norm_adaptativebin_erode")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    processing.run(loader, output_dir)
    processing.show_frame(20)
    processing.show_video()
