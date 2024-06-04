"""
Gaussian blur
Normalization
Binarisation using an adaptative threshold
Moprhological erosion
"""

import cv2 as cv

import utils


p = utils.Pipeline()

p.add("blur", lambda img: cv.GaussianBlur(img, (9, 9), 0))
p.add("norm", lambda img: cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
# p.add("binary", lambda img: cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2))
p.add("binary", lambda img: cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2))
p.add("morph", lambda img: cv.erode(img, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))))
# p.add("morph", lambda img: cv.morphologyEx(img, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)), iterations=2))


if __name__ == '__main__':
    from pathlib import Path
    import image_loader
    
    dir_path = Path("imgs", "vertical")
    # dir_path = Path("imgs", "diagonal")
    loader = image_loader.ImageLoaderColorConverter(dir_path, cv.COLOR_RGB2GRAY)

    p.run(loader)
    p.show_samples(20)
    p.show_videos()
