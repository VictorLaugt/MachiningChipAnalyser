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

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "norm_tresh_morphedge")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    pipeline.run(loader, output_dir)
    pipeline.show_frame(20)
    pipeline.show_video()
