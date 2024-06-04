"""
Canny edge detection
"""

import cv2 as cv

import utils


p = utils.Pipeline()
p.add("edge", lambda img: cv.Canny(img, 100, 200))


if __name__ == '__main__':
    from pathlib import Path
    import image_loader

    dir_path = Path("imgs", "vertical")
    # dir_path = Path("imgs", "diagonal")
    loader = image_loader.ImageLoaderColorConverter(dir_path, cv.COLOR_RGB2GRAY)

    p.run(loader)
    p.show_samples(43)
    p.show_videos()
