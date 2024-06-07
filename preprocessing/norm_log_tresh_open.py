"""
Normalization
LoG edge extraction (with different matrix coefficients for contact and spikes detection)
Threshold binarization
Morphological opening and closing (for contact detection only)
"""

import numpy as np
import cv2 as cv

import utils


log_kernel = np.array([
    [0, 1, 1, 2, 2, 2, 1, 1, 0],
    [1, 2, 4, 5, 5, 5, 4, 2, 1],
    [1, 4, 5, 3, 0, 3, 5, 4, 1],
    [2, 5, 3, -12, -24, -12, 3, 5, 2],
    [2, 5, 0, -24, -40, -24, 0, 5, 2],
    [2, 5, 3, -12, -24, -12, 3, 5, 2],
    [1, 4, 5, 3, 0, 3, 5, 4, 1],
    [1, 2, 4, 5, 5, 5, 4, 2, 1],
    [0, 1, 1, 2, 2, 2, 1, 1, 0]
])


# contact detection
processing_contact = utils.DagProcess()
processing_contact.add("norm", lambda img: cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
processing_contact.add("contact_edge", lambda img: cv.filter2D(img, -1, 2 * log_kernel))
processing_contact.add("contact_binary", lambda img: cv.threshold(img, 245, 255, cv.THRESH_BINARY)[1])
processing_contact.add("contact_open", lambda img: cv.morphologyEx(img, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))))
processing_contact.add("contact_close", lambda img: cv.morphologyEx(img, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=5))


# spike detection
processing_spikes = utils.DagProcess()
processing_spikes.add("norm", lambda img: cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
processing_spikes.add("spikes_edge", lambda img: cv.filter2D(img, -1, (1/3) * log_kernel))
processing_spikes.add("spikes_binary", lambda img: cv.threshold(img, 245, 255, cv.THRESH_BINARY)[1])


if __name__ == '__main__':
    from pathlib import Path
    import image_loader

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "norm_log_thresh_open")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    processing_contact.run(loader, output_dir)
    processing_spikes.run(loader, output_dir)

    processing_contact.show_frame(20)
    processing_contact.show_video()

    processing_spikes.show_frame(20)
    processing_spikes.show_video()
