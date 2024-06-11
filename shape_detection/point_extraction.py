import sys

import numpy as np
import cv2 as cv

import geometry


def draw_line(img, rho, xn, yn, color, thickness):
    """Draw on img the line whose polar parameters are
    (rho, theta), with xn = cos(theta), yn = sin(theta).
    """
    x0, y0 = rho * xn, rho * yn
    x1, y1 = int(x0 - 2000 * yn), int(y0 + 2000 * xn)
    x2, y2 = int(x0 + 2000 * yn), int(y0 - 2000 * xn)
    cv.line(img, (x1, y1), (x2, y2), color, thickness)


def locate_base_and_tool(binary_img):
    """Compute line parameters for base and tool."""
    lines = cv.HoughLines(binary_img, 1, np.pi/180, 100)
    if lines is None or len(lines) < 2:
        raise ValueError("Warning !: line not found")

    rho0, theta0 = geometry.positive_rho(*lines[0, 0, :])
    rho1, theta1 = geometry.positive_rho(*lines[1, 0, :])

    xn0, yn0 = np.cos(theta0), np.sin(theta0)
    xn1, yn1 = np.cos(theta1), np.sin(theta1)

    return (rho0, xn0, yn0), (rho1, xn1, yn1)


def extract_points(binary_img):
    """Return coordinates of points between base and tool, and line parameters
    for base and tool.
    """
    (rho0, xn0, yn0), (rho1, xn1, yn1) = locate_base_and_tool(binary_img)

    contours, _hierarchy = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    points = np.vstack(contours)
    x, y = points[:, 0, 0], points[:, 0, 1]

    mask = (
        (xn0*x + yn0*y - rho0 + 10 <= 0) &
        (xn1*x + yn1*y - rho1 + 10 <= 0)
    ).flatten()
    filtered_points = points[mask]

    return filtered_points, (rho0, xn0, yn0), (rho1, xn1, yn1)


def render_point_extraction(binary_img):
    render = np.zeros_like(binary_img)

    points, line0, line1 = extract_points(binary_img)

    x, y = points[:, 0, 0], points[:, 0, 1]
    render[y, x] = 255

    draw_line(render, *line0, color=127, thickness=1)
    draw_line(render, *line1, color=127, thickness=1)

    return render


if __name__ == '__main__':
    import image_loader
    from pathlib import Path

    import preprocessing.log_tresh_blobfilter_erode

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("pointextraction", render_point_extraction)

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "lines")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    processing.run(loader, output_dir)
    processing.show_frame(21)
    processing.compare_videos(("input", "pointextraction"))
