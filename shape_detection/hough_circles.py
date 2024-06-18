import sys

import geometry
from shape_detection.chip_extraction import extract_main_features

import numpy as np
import cv2 as cv


def draw_chip_curve(mask, hull_points):
    margin = 5

    h, w = mask.shape
    pts = hull_points.reshape(-1, 2)

    for i in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[i], pts[i+1]
        if (
                margin < x2 < w-margin and margin < y2 < h-margin and
                margin < x1 < w-margin and margin < y1 < h-margin
        ):
            cv.line(mask, (x1, y1), (x2, y2), 255, 5)


def extract_circles(binary_img):
    main_ft = extract_main_features(binary_img)

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))

    x_min = chip_pts[chip_pts[:, :, 0].argmin(), 0, 0]
    x_max = chip_pts[chip_pts[:, :, 1].argmax(), 0, 1]

    circles = cv.HoughCircles(
        binary_img,
        cv.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30,
        minRadius=int(0.5 * (x_max - x_min)),
        maxRadius=int(2.0 * (x_max - x_min))
    )

    if circles is None:
        print("Warning !: circle not found", file=sys.stderr)

    return circles, main_ft.base_line, main_ft.tool_line


def render_circles(binary_img, render=None):
    if render is None:
        render = np.zeros_like(binary_img)
    else:
        render = render.copy()

    circles, base_line, tool_line = extract_circles(binary_img)

    geometry.draw_line(render, base_line, color=127, thickness=1)
    geometry.draw_line(render, tool_line, color=127, thickness=1)
    if circles is not None:
        for x_center, y_center, radius in circles[0, :, :]:
            cv.circle(render, (int(x_center), int(y_center)), int(radius), 127, 1)

    return render


if __name__ == '__main__':
    from pathlib import Path

    import image_loader
    import preprocessing.log_tresh_blobfilter_erode

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("circles", render_circles)

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "circles")
    loader = image_loader.ImageLoader(input_dir)

    processing.run(loader, output_dir)
    processing.compare_frames(25, ("morph", "circles"))
    processing.compare_videos(("morph", "circles"))


# if __name__ == '__main__':
#     img = cv.imread('preprocessed.png', cv.IMREAD_GRAYSCALE)
#     # img = cv.imread('chipcurve.png', cv.IMREAD_GRAYSCALE)

#     inp = img.copy()
#     inp[:, :500] = 0
#     inp[:, 966:] = 0
#     inp[385:, :] = 0

#     circles = cv.HoughCircles(
#         inp,
#         cv.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30,
#         minRadius=0,
#         maxRadius=0
#     )

#     render = inp.copy()
#     # for c in circles[0, :10, :]:
#     for c in circles[0, :, :]:
#         cv.circle(render, (int(c[0]), int(c[1])), int(c[2]), 127, 1)

#     cv.imshow('circles', render)
#     while cv.waitKey(30) != 113:
#         pass
#     cv.destroyAllWindows()


#     def show_circle(img, circle):
#         img = img.copy()
#         cv.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), 127, 1)
#         cv.imshow('circle', img)
#         while cv.waitKey(30) != 113:
#             pass
#         cv.destroyAllWindows()

