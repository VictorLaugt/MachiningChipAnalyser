from pathlib import Path

import cv2 as cv
import numpy as np

import video


h, w = 200, 200

def above_line(points, rho, xn, yn, min_distance):
    x, y = points[:, :, 0], points[:, :, 1]
    mask = (xn*x + yn*y - rho >= min_distance).flatten()
    return points[mask]

def under_line(points, rho, xn, yn, min_distance):
    x, y = points[:, :, 0], points[:, :, 1]
    mask = (xn*x + yn*y - rho <= min_distance).flatten()
    return points[mask]

def draw(img, rho, xn, yn, color, thickness):
    x0, y0 = rho * xn, rho * yn
    x1, y1 = int(x0 - 1000 * yn), int(y0 + 1000 * xn)
    x2, y2 = int(x0 + 1000 * yn), int(y0 - 1000 * xn)
    cv.line(img, (x1, y1), (x2, y2), color, thickness)
    cv.circle(img, (int(x0), int(y0)), 2*thickness, color, -1)

seq = []

every_points = np.stack(np.indices((h, w)), axis=-1).reshape(-1, 1, 2)
rho = 150
# for i, theta in enumerate(np.linspace(0, 2*np.pi, 100)):
#     img = np.zeros((h, w), dtype=np.uint8)
#     xn = np.cos(theta)
#     yn = np.sin(theta)

#     pts = under_line(every_points, rho, xn, yn, min_distance=5)
#     x, y = pts[:, 0, 0].flatten(), pts[:, 0, 1].flatten()
#     img[y, x] = 255
#     draw(img, rho, xn, yn, 127, 5)

#     seq.append(img)

#     print(f"frame {i}: {theta = }")


# video.create_from_gray(seq, Path("results", "polar_line_experiment.avi"))
# video.play(Path("results", "polar_line_experiment.avi"))


theta0, theta1 = 0.0, np.pi/2
xn0, yn0 = np.cos(theta0), np.sin(theta0)
xn1, yn1 = np.cos(theta1), np.sin(theta1)

img = np.zeros((h, w), dtype=np.uint8)

pts = every_points.copy()
pts = under_line(pts, rho, xn1, yn1, 5)
pts = under_line(pts, rho, xn0, yn0, 5)

x, y = pts[:, 0, 0].flatten(), pts[:, 0, 1].flatten()
img[y, x] = 255


cv.imshow('img', img)
while cv.waitKey(30) != 113:
    pass
cv.destroyAllWindows()
