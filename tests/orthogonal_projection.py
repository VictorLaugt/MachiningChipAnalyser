import numpy as np
import cv2 as cv
import geometry


img = np.zeros((100, 100), dtype=np.uint8)

line = (50, np.cos(np.pi/3), np.sin(np.pi/3))
pt = (74, 32)
projected_pt = geometry.orthogonal_projection(*pt, line)
int_projected_pt = (int(projected_pt[0]), int(projected_pt[1]))

geometry.draw_line(img, line, 255, 1)
cv.circle(img, pt, 3, 127, -1)
cv.circle(img, int_projected_pt, 3, 127, -1)

cv.imshow("Orthogonal projection", img)
while cv.waitKey(30) != 113:
    pass
cv.destroyAllWindows()
