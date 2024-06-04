from pathlib import Path

import numpy as np
import cv2 as cv


def extract_chip_curve(binary):
    extracted_shape = binary.copy()
    contours, _hierarchy = cv.findContours(binary[:, :967], cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    every_contours = np.vstack(contours)  # ~ (n, 1, 2)
    hull = cv.convexHull(every_contours)  # ~ (p, 1, 2)

    cv.drawContours(extracted_shape, contours, -1, 255, 2)
    cv.drawContours(extracted_shape, (hull,), 0, 127, 0)

    return extracted_shape


if __name__ == '__main__':
    import utils
    import image_loader

    from preprocessing.log_tresh_erode import pipeline as preprocessing

    # binary = cv.cvtColor(cv.imread('morph2.png'), cv.COLOR_RGB2GRAY)

    pipeline = utils.Pipeline()
    pipeline.add("chip_curve", extract_chip_curve)

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "chip_curve")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    preprocessing.run(loader)

    pipeline.run(preprocessing.get_output(), output_dir)
    pipeline.show_samples(20)
    pipeline.show_videos()




# cv.imshow('img', binary)
# cv.imshow('extracted_contours', extracted_shape)
# while cv.waitKey(30) != 113:
#     pass
# cv.destroyAllWindows()

