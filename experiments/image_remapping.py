from pathlib import Path
import cv2 as cv
import numpy as np

from triangle_to_rectangle_deformation import tri_to_rect_mapping, tri_to_rect_unmapping

black = (0, 0, 0)
red = (0, 0, 255)

img = cv.imread(str(Path(__file__).resolve().parent.joinpath('mountain.jpg')))
h, w = img.shape[:2]
img[::25, :] = img[:, ::25] = black

# controls the deformation
a, b, c = (0, h//3), (w, h), (w//2, 0)
interpolation_flag = cv.INTER_LINEAR

# deforms
dst_idx = np.dstack(np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)))
src_idx = tri_to_rect_unmapping(dst_idx, a, b, c)
map_x = src_idx[:, :, 0].astype(np.float32)
map_y = src_idx[:, :, 1].astype(np.float32)
deformed_img = cv.remap(img, map_x, map_y, interpolation_flag)

# restores
src_idx = np.dstack(np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)))
dst_idx = tri_to_rect_mapping(src_idx, a, b, c)
map_x = dst_idx[:, :, 0].astype(np.float32)
map_y = dst_idx[:, :, 1].astype(np.float32)
restaured_img = cv.remap(deformed_img, map_x, map_y, interpolation_flag)

cv.circle(img, a, 5, red, -1)
cv.circle(img, b, 5, red, -1)
cv.line(img, a, b, red, 2)
cv.circle(img, c, 5, red, -1)
cv.imshow('img', img)
cv.imshow('deformed_img', deformed_img)
cv.imshow('restaured_img', restaured_img)
while cv.waitKey(0) != ord('q'):
    pass
cv.destroyAllWindows()
