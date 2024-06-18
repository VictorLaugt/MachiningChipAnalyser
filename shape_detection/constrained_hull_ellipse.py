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


def extract_chip_curve(precise, rough):
    main_ft = extract_main_features(precise)
    h, w = precise.shape

    contours, _ = cv.findContours(precise, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))

    # compute the convex hull and constrain it to cross an anchor point
    x_min = chip_pts[chip_pts[:, :, 0].argmin(), 0, 0]
    anchor = np.array([[[x_min, h-1]]])
    hull_points = cv.convexHull(np.vstack((chip_pts, anchor)))
    first_point_index = np.where(
        (hull_points[:, 0, 0] == anchor[0, 0, 0]) &
        (hull_points[:, 0, 1] == anchor[0, 0, 1])
    )[0][0]
    hull_points = np.roll(hull_points, -first_point_index, axis=0)

    # remove from the convex hull points near the tool and the base
    chip_curve_keys = geometry.under_lines(hull_points, (main_ft.base_line, main_ft.tool_line), (20, 20))

    # extract points of the chip curve
    chip_curve_mask = np.zeros((h, w), dtype=np.uint8)
    draw_chip_curve(chip_curve_mask, chip_curve_keys)
    y, x = np.nonzero(rough & chip_curve_mask)
    chip_curve_points = np.stack((x, y), axis=1).reshape(-1, 1, 2)

    if len(chip_curve_points) >= 5:
        ellipse = cv.fitEllipse(chip_curve_points)
    else:
        ellipse = None
        print(f"Warning !: Not enough point to fit an ellipse ({len(chip_curve_points)} points)", file=sys.stderr)

    return ellipse, main_ft.base_line, main_ft.tool_line


def render_chip_curve(precise, rough, render=None):
    if render is None:
        render = np.zeros_like(precise)
    else:
        render = render.copy()

    ellipse, base_line, tool_line = extract_chip_curve(precise, rough)

    geometry.draw_line(render, base_line, color=127, thickness=1)
    geometry.draw_line(render, tool_line, color=127, thickness=1)
    if ellipse is not None:
        cv.ellipse(render, ellipse, 127, 1)

    return render


if __name__ == '__main__':
    from pathlib import Path

    import image_loader
    import preprocessing.log_tresh_blobfilter_erode

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("chipcurve", render_chip_curve, ("morph", "blobfilter"))

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "chipcurve")
    loader = image_loader.ImageLoader(input_dir)

    processing.run(loader, output_dir)
    processing.compare_frames(25, ("morph", "chipcurve"))
    processing.compare_videos(("input", "chipcurve"))


# if __name__ == '__main__':
#     img = cv.cvtColor(cv.imread('preprocessed.png'), cv.COLOR_RGB2GRAY)
#     extracted = extract_chip_curve(img, img)

#     cv.imshow('extracted', extracted)
#     while cv.waitKey(30) != 113:
#         pass
#     cv.destroyAllWindows()
