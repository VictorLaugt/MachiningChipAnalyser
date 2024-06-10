import sys
from pathlib import Path

import numpy as np
import cv2 as cv


def above_line(points, a, b, c, min_distance):
    """Filter the points to keep those which verify a*x + b*y + c >= min_distance"""
    x, y = points[:, :, 0], points[:, :, 1]
    mask = (a*x + b*y + c >= min_distance).flatten()
    return points[mask]


def draw_lines(img, points, color, thickness):
    points = points.reshape(-1, 2)
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        cv.line(img, start_point, end_point, color, thickness)


# def draw_chip_curve_mask(mask, hull_points):
#     pts = hull_points.reshape(-1, 2)

#     argmin = 0
#     y_curr = mask.shape[0]
#     for i in range(len(pts) - 1):
#         y = pts[i, 1]
#         if y < y_curr:
#             y_curr = y
#             argmin = i

#     if pts[argmin-1, 0] <= pts[argmin+1, 0]:
#         argmin -= 1

#     for i in range(0, argmin):
#         cv.line(mask, pts[i], pts[i+1], 255, 5)
#     for i in range(argmin+1, len(pts) - 1):
#         cv.line(mask, pts[i], pts[i+1], 255, 5)


def draw_chip_curve_mask(mask, hull_points):
    h, w = mask.shape
    pts = hull_points.reshape(-1, 2)

    def is_outside(point):
        return not (10 < point[0] < w-10 and 10 < point[1] < h-10)

    def draw(start_index):
        i = start_index
        for i in range(start_index, len(pts) - 1):
            a, b = pts[i], pts[i+1]
            if is_outside(b):
                break
            cv.line(mask, a, b, 255, 5)
        return i

    def skip(index):
        while index < len(pts) and is_outside(pts[index]):
            index += 1
        return index

    start = 0
    for _ in range(2):
        end = draw(start)
        start = skip(end + 1)
    draw(start)


def extract_chip_curve(precise, rough):
    h, w = precise.shape

    contours, _hierarchy = cv.findContours(precise, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    points = np.vstack(contours)  # ~ (n, 1, 2)

    # MOCK: remove the tool and the base
    points = above_line(points, a=0, b=-1, c=385, min_distance=5)  # above the base
    points = above_line(points, a=-1, b=0, c=967, min_distance=5)  # at the left of the tool

    # compute the convex hull and constrain it to cross an anchor point
    x_min = points[points[:, :, 0].argmin(), 0, 0]
    anchor = np.array([[[x_min, h-1]]])
    points = np.vstack((points, anchor))
    hull_points = cv.convexHull(points)  # ~ (p, 1, 2)

    # remove points of the convex hull near the tool, the base, and the up border
    hull_points = above_line(hull_points, a=0, b=-1, c=385, min_distance=20)
    hull_points = above_line(hull_points, a=-1, b=0, c=967, min_distance=10)
    # hull_points = above_line(hull_points, a=0, b=1, c=0, min_distance=5)

    # extract points near the hull
    mask = np.zeros((h, w), dtype=np.uint8)
    draw_chip_curve_mask(mask, hull_points)

    y, x = np.nonzero(rough & mask)
    # y, x = np.nonzero(precise & mask) # TODO: test what is the best between extracting from rough or precise
    chip_curve_points = np.stack((x, y), axis=1).reshape(-1, 1, 2)

    if len(chip_curve_points) >= 5:
        ellipse = cv.fitEllipse(chip_curve_points)
    else:
        print("Warning !: Not enough point to fit an ellipse", file=sys.stderr)

    # display
    to_display = np.zeros((h, w), dtype=np.uint8)
    to_display[y, x] = 255
    # for pt in hull_points.reshape(-1, 2):
    #     cv.circle(to_display, pt, 5, 127, -1)
    if len(chip_curve_points) >= 5:
        cv.ellipse(to_display, ellipse, 127, 1)

    return to_display


if __name__ == '__main__':
    import image_loader

    import preprocessing.log_tresh_blobfilter_erode

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("chipcurve", extract_chip_curve, ("erode", "clean"))

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "chipcurve")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    processing.run(loader, output_dir)
    # processing.show_frame(25)
    processing.compare_frames(25, ("erode", "chipcurve"), horizontal=True)
    # processing.show_video()
    processing.compare_videos(("erode", "chipcurve"), horizontal=True)


# if __name__ == '__main__':
#     img = cv.cvtColor(cv.imread('preprocessed.png'), cv.COLOR_RGB2GRAY)
#     extracted = extract_chip_curve(img, img)

#     cv.imshow('extracted', extracted)
#     while cv.waitKey(30) != 113:
#         pass
#     cv.destroyAllWindows()
