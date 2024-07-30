from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable
    from type_hints import Image, OpenCVIntArray, IntPtArray, IntArray
    from img_loader import AbstractImageLoader
    from features_main import MainFeatures
    import numpy as np
    BinImage = np.ndarray[tuple[int, int], np.uint8]

import numpy as np
import cv2 as cv

from preproc import preprocess
from features_main import extract_main_features, best_base_line
import geometry


def wait_q_pressed() -> None:
    while cv.waitKey(0) != 113:
        pass
    cv.destroyAllWindows()


def mean_image(images: Iterable[Image]) -> Image:
    image_iterator = iter(images)
    accumulator = next(image_iterator).astype(np.uint64)
    image_count = 1
    for img in image_iterator:
        accumulator += img
        image_count += 1
    return (accumulator / image_count).astype(np.uint8)


def extract_tool_penetration(loader: AbstractImageLoader) -> None:
    first_image = next(iter(loader))
    main_ft = extract_main_features(preprocess(first_image))

    mean_img = mean_image(preprocess(img) for img in loader)
    thresh = np.median(mean_img[mean_img > 100])
    print(f"{thresh = }")
    bin_img = cv.threshold(mean_img, thresh, 255, cv.THRESH_BINARY)[1]

    y, x = np.nonzero(bin_img)
    pts: OpenCVIntArray = np.column_stack((x, y)).reshape(-1, 1, 2)

    above_pts = geometry.above_lines(pts, (main_ft.base_line, main_ft.tool_line), (-5, -5))
    tip_bin_img = np.zeros_like(bin_img)
    tip_bin_img[above_pts[:, 0, 1], above_pts[:, 0, 0]] = 255

    lines = cv.HoughLines(tip_bin_img, 1, np.pi/180, 1)
    if lines is None or len(lines) < 1:
        raise ValueError("line not found")
    print(f"Found {len(lines)} lines")

    rho_tip, theta_tip = geometry.standard_polar_param(*best_base_line(lines))
    xn_tip, yn_tip = np.cos(theta_tip), np.sin(theta_tip)
    geometry.draw_line(tip_bin_img, (rho_tip, xn_tip, yn_tip), 127, 1)

    cv.imshow("tip_bin_img", tip_bin_img)
    wait_q_pressed()
