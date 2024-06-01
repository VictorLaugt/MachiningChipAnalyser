"""
LoG edge extraction -> Threshold binarization -> Morphological operations
"""

from pathlib import Path

import numpy as np
import cv2 as cv

import image_loader
import video


dir_path = Path("imgs", "vertical")
# dir_path = Path("imgs", "diagonal")
loader = image_loader.ImageLoaderColorConverter(dir_path, cv.COLOR_RGB2GRAY)


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


def morph_succession(img, succession):
    for operation, structure in succession:
        img = cv.morphologyEx(img, operation, structure)
    return img


#  detection
log_kernel_ = 2 * log_kernel

edge = [cv.filter2D(img, -1, log_kernel_) for img in loader]

norm_edge = [cv.normalize(img, None, 0, 255, cv.NORM_MINMAX) for img in edge]

binary = [cv.threshold(img, 240, 255, cv.THRESH_BINARY)[1] for img in norm_edge]

# TODO: try different morphological succession
morph = [
    morph_succession(
        img,
        [
            (cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))),
            # (cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS, (2, 2))),
        ]
    )
    for img in binary
]

# morph = [
#     cv.morphologyEx(
#         img,
#         cv.MORPH_OPEN,
#         cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)),
#         iterations=2
#     )
#     for img in binary
# ]



# image display
sample_idx = 20
cv.imshow("original", loader[sample_idx])

cv.imshow("norm edge", norm_edge[sample_idx])
cv.imshow("binary", binary[sample_idx])
cv.imshow("morph", morph[sample_idx])

while cv.waitKey(30) != 113:
    pass
cv.destroyAllWindows()


# video display
video.create_from_gray(loader, "original.avi")

video.create_from_gray(norm_edge, "meth3_norm_edge.avi")
video.create_from_gray(binary, "meth3_binary.avi")
video.create_from_gray(morph, "meth3_morph.avi")

video.play("original.avi")
video.play("meth3_norm_edge.avi")
video.play("meth3_binary.avi")
video.play("meth3_morph.avi")
