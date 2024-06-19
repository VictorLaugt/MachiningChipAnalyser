from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    from typing import Callable, TypeVar
    Image = TypeVar("Image", bound=np.ndarray)

import numpy as np
import cv2 as cv

import connected_components


log_kernel = 2 * np.array([
    [0, 1, 1, 2, 2, 2, 1, 1, 0],
    [1, 2, 4, 5, 5, 5, 4, 2, 1],
    [1, 4, 5, 3, 0, 3, 5, 4, 1],
    [2, 5, 3, -12, -24, -12, 3, 5, 2],
    [2, 5, 0, -24, -40, -24, 0, 5, 2],
    [2, 5, 3, -12, -24, -12, 3, 5, 2],
    [1, 2, 4, 5, 5, 5, 4, 2, 1],
    [1, 4, 5, 3, 0, 3, 5, 4, 1],
    [0, 1, 1, 2, 2, 2, 1, 1, 0]
])


def processing(img: Image) -> Image:
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    edge_img = cv.filter2D(gray_img, -1, log_kernel)
    binary_img = cv.threshold(edge_img, 240, 255, cv.THRESH_BINARY)[1]
    blobfilter_img = connected_components.remove_small_components(binary_img, min_area=20)
    morph_img = cv.erode(blobfilter_img, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)), iterations=2)
    return morph_img


def real_time_processing(stream: cv.VideoCapture, processing: Callable[[Image], Image]) -> None:
    while True:
        status, img = stream.read()
        if not status: # reached end of file
            break
        cv.imshow("processed image", processing(img))
        if cv.waitKey(1) == 113: # user exits by pressing 'q'
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    from pathlib import Path

    cam = cv.VideoCapture(0)
    # cam = cv.VideoCapture(str(Path("experiments", "input.avi")))
    real_time_processing(cam, processing)
    cam.release()
