import numpy as np
import cv2 as cv

import geometry


def hough_point_set(img, pts):
    lines = cv.HoughLinesPointSet(
        pts,
        lines_max=10,  # maximum number of returned line
        threshold=1,   # minimum number of votes for a line to be returned
        min_rho=-900,
        max_rho=900,
        rho_step=1,
        min_theta=0,
        max_theta=np.pi,
        theta_step=np.pi/180
    )
    if lines is None:
        raise ValueError("no line found")

    print(f"Found {len(lines)} lines")
    for _votes, rho, theta in lines[:20, 0, :]:
        xn, yn = np.cos(theta), np.sin(theta)
        geometry.draw_line(img, (rho, xn, yn), 127, 1)


def hough_probabilistic(img):
    lines = cv.HoughLinesP(
        img,
        rho=1,            # rho resolution
        theta=np.pi/180,  # theta resolution
        threshold=1,      # minimum number of votes for a segment to be returned
        minLineLength=35, # minimum length of the returned segments
        maxLineGap=100    # maximum allowed gap between points on the same line to link them
    )
    if lines is None:
        raise ValueError("no line found")

    # img[:] = 0

    print(f"Found {len(lines)} lines")
    for x0, y0, x1, y1 in lines[:20, 0, :]:
        cv.line(img, (x0, y0), (x1, y1), 127, 1)



def hough(img):
    lines = cv.HoughLines(
        img,
        rho=1,            # rho resolution
        theta=np.pi/180,  # theta resolution
        threshold=1       # minimum number of votes for a line to be returned
    )
    if lines is None:
        raise ValueError("no line found")

    print(f"Found {len(lines)} lines")
    for rho, theta in lines[:20, 0, :]:
        xn, yn = np.cos(theta), np.sin(theta)
        geometry.draw_line(img, (rho, xn, yn), 127, 1)



if __name__ == '__main__':
    from pathlib import Path

    # n_pts = 200
    # x = np.random.choice(range(900), n_pts, replace=False)
    # y = np.random.randint(250, 260, (n_pts))
    # pts = np.column_stack((x, y)).reshape(-1, 1, 2)

    # img = np.zeros((500, 900), dtype=np.uint8)
    # img[y, x] = 255


    color_img = cv.imread(str(Path('experiments', 'chipinside.png')))
    img = np.ascontiguousarray(color_img[:, :, 2])
    y, x = np.nonzero(img)
    pts = np.column_stack((x, y)).reshape(-1, 1, 2)


    # hough_point_set(img, pts)
    hough_probabilistic(img)
    # hough(img)

    cv.imshow('img', img)
    while cv.waitKey(0) != 113:
        pass
    cv.destroyAllWindows()
