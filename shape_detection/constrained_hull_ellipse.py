import sys

from shape_detection.chip_extraction import (
    extract_chip_points, filter_between_base_tool, draw_line
)

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
    h, w = precise.shape

    points, base_line, tool_line = extract_chip_points(precise)

    # compute the convex hull and constrain it to cross an anchor point
    x_min = points[points[:, :, 0].argmin(), 0, 0]
    anchor = np.array([[[x_min, h-1]]])
    hull_points = cv.convexHull(np.vstack((points, anchor)))
    first_point_index = np.where(
        (hull_points[:, 0, 0] == anchor[0, 0, 0]) &
        (hull_points[:, 0, 1] == anchor[0, 0, 1])
    )[0][0]
    hull_points = np.roll(hull_points, -first_point_index, axis=0)

    # remove from the convex hull points near the tool and the base
    chip_curve_keys = filter_between_base_tool(hull_points, base_line, tool_line, 20, 20)

    # extract points of the chip curve
    chip_curve_mask = np.zeros((h, w), dtype=np.uint8)
    draw_chip_curve(chip_curve_mask, chip_curve_keys)
    y, x = np.nonzero(rough & chip_curve_mask)
    chip_curve_points = np.stack((x, y), axis=1).reshape(-1, 1, 2)

    if len(chip_curve_points) >= 5:
        ellipse = cv.fitEllipse(chip_curve_points)
    else:
        print(f"Warning !: Not enough point to fit an ellipse ({len(chip_curve_points)} points)", file=sys.stderr)

    # rendering
    render = precise.copy()
    draw_line(render, *base_line, 127, 1)
    draw_line(render, *tool_line, 127, 1)
    for pt in chip_curve_keys.reshape(-1, 2):
        cv.circle(render, pt, 5, 127, -1)
    # cv.drawContours(render, (hull_points,), 0, 127, 0)
    if len(chip_curve_points) >= 5:
        cv.ellipse(render, ellipse, 127, 1)

    return render


if __name__ == '__main__':
    from pathlib import Path

    import image_loader
    import preprocessing.log_tresh_blobfilter_erode

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("chipcurve", extract_chip_curve, ("erode", "clean"))

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "chipcurve")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    processing.run(loader, output_dir)
    processing.compare_frames(25, ("erode", "chipcurve"))
    processing.compare_videos(("erode", "chipcurve"))


# if __name__ == '__main__':
#     img = cv.cvtColor(cv.imread('preprocessed.png'), cv.COLOR_RGB2GRAY)
#     extracted = extract_chip_curve(img, img)

#     cv.imshow('extracted', extracted)
#     while cv.waitKey(30) != 113:
#         pass
#     cv.destroyAllWindows()
