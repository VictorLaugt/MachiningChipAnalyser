import numpy as np
import cv2 as cv
import geometry


img = np.zeros((300, 300), dtype=np.uint8)

line = (150, np.cos(np.pi/3), np.sin(np.pi/3))
pts = np.array([
    [194, 132],
    [13, 214],
    [16, 61],
    [168, 155],
], dtype=np.int32).reshape(-1, 1, 2)

idx, distance = geometry.line_furthest_point(pts, line)

for i in range(len(pts)):
    cv.circle(img, tuple(pts[i, 0, :]), 3, 127 // 2, -1)
cv.circle(img, tuple(pts[idx, 0, :]), 3, 127, -1)
geometry.draw_line(img, line, 255, 1)
print(f"{distance = }")

cv.imshow("Points", img)
while cv.waitKey(30) != 113:
    pass
cv.destroyAllWindows()
