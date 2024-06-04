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

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "tresh_morphedge")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    pipeline.run(loader, output_dir)
    pipeline.show_samples(40)
    pipeline.show_videos()
