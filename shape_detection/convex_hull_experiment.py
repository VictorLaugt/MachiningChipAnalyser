from pathlib import Path

import numpy as np
import cv2 as cv


binary = cv.cvtColor(cv.imread('morph2.png'), cv.COLOR_RGB2GRAY)

contours, _hierarchy = cv.findContours(binary[:, :967], cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
every_contours = np.vstack(contours)  # ~ (n, 1, 2)
hull = cv.convexHull(every_contours, clockwise=True)  # ~ (p, 1, 2)


for i in range(3, min(10, len(hull))):
    hull_draw = binary.copy()
    cv.drawContours(hull_draw, (hull[:i+1],), 0, 127, 3)
    cv.imshow(f'hull[:{i+1}]', hull_draw)

while cv.waitKey(30) != 113:
    pass
cv.destroyAllWindows()

