from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    from type_hints import GrayImage, ColorImage, OpenCVIntArray, IntPt, IntPtArray, IntArray, FloatPtArray, FloatArray
    from features_main import MainFeatures

import numpy as np
import cv2 as cv
from skimage.draw import line

import colors


class InsideFeatures:
    __slots__ = (
        "noised_inside_contour_pts",  # type: IntPtArray
        "inside_contour_pts",         # type: IntPtArray
        "thickness"                   # type: FloatArray
    )


def rasterized_line(p0: IntPt, p1: IntPt, img_height: int, img_width: int) -> tuple[IntArray, IntArray]:
    """Return the rasterization of the line segment defined by two points.

    Parameters
    ----------
    p0, p1: IntPt
        Points defining the segment.
    img_height, img_width: int
        Height and width of the image.

    Returns
    -------
    raster_x, raster_y: (n,)-array of int
        Respectively the x and y coordinates of the image pixels crossed by the
        line segment (p0, p1). 0 <= x < img_width and 0 <= y < img_height.
    """
    line_x, line_y = line(*p0, *p1)
    inside_mask = (0 <= line_x) & (line_x < img_width) & (0 <= line_y) & (line_y < img_height)
    raster_x, raster_y = line_x[inside_mask], line_y[inside_mask]
    return raster_x, raster_y


def compute_bisectors(
    chip_curve_pts: OpenCVIntArray,
    indirect_chip_rotation: bool
) -> FloatPtArray:
    """Return the unit vectors bisecting the edges of the chip curve.

    Vector bisectors[i] is bisecting the two adjacent edges
    (chip_curve_pts[i-1], chip_curve[i]) and (chip_curve[i], chip_curve[i+1]).
    Vector bisectors[0] is normal to the edge (chip_curve_pts[0], chip_curve_pts[1]).
    Vector bisectors[-1] is normal to the edge (chip_curve_pts[-2], chip_curve_pts[-1]).

    Parameters
    ----------
    chip_curve_pts: (n, 1, 2)-array of int
        Points of the chip convex hull that describes the chip curve.
    indirect_chip_rotation: bool
        True indicates the chip spins in the indirect direction of rotation.

    Returns
    -------
    bisectors: (n, 2)-array of float
        bisector unit vectors
    """
    pts = chip_curve_pts.reshape(-1, 2)
    bisectors = np.zeros_like(pts, dtype=np.float32)

    u = pts[:-2] - pts[1:-1]
    v = pts[2:] - pts[1:-1]
    w = v * ((np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1))).reshape(-1, 1)

    # numerical instability correction if the angle between u and v is greater than pi/2
    stable = (np.sum(u*v, axis=1) > 0)
    unstable = ~stable

    bisectors[1:-1][stable] = u[stable] + w[stable]
    normal = u[unstable] - w[unstable]

    normal_first = pts[0] - pts[1]
    normal_last = pts[-2] - pts[-1]
    if indirect_chip_rotation:
        bisectors[1:-1][unstable] = np.column_stack((-normal[:, 1], normal[:, 0]))
        bisectors[0] = (-normal_first[1], normal_first[0])
        bisectors[-1] = (-normal_last[1], normal_last[0])
    else:
        bisectors[1:-1][unstable] = np.column_stack((normal[:, 1], -normal[:, 0]))
        bisectors[0] = (normal_first[1], -normal_first[0])
        bisectors[-1] = (normal_last[1], -normal_last[0])

    return bisectors / np.linalg.norm(bisectors, axis=1).reshape(-1, 1)


def remove_crossed_quadrilaterals(
    out_curve: IntPtArray,
    in_curve: IntPtArray
) -> tuple[IntPtArray, IntPtArray]:
    """Return the inside and outside curves of the chip without crossed
    quadrilaterals.

    (filt_out_curve[i], filt_in_curve[i], fil_in_curve[i+1], filt_out_curve[i+1])
    is not a crossed quadrilateral.

    Parameters
    ----------
    out_curve, in_curve: (n, 2)-array of int
        Respectively the outside and inside curves of the chip. n >= 2.

    Returns
    -------
    filt_out_curve, filt_in_curve: (m, 2)-array of int
        Respectively the outside and inside curves of the chip without crossed
        quadrilaterals. 2 <= m < n.
    """
    while True:
        o1, i1 = out_curve[:-1], in_curve[:-1]
        o2, i2 = out_curve[1:], in_curve[1:]
        not_crossed_mask = np.ones(len(out_curve), dtype=bool)
        not_crossed_mask[:-1] = ((np.cross(i1-o1, i2-i1) * np.cross(o2-i2, o1-o2)) > 0)

        if not_crossed_mask.all() or len(out_curve) == 2:
            return out_curve, in_curve
        else:
            out_curve = out_curve[not_crossed_mask]
            in_curve = in_curve[not_crossed_mask]


def find_inside_contour(
    chip_bin_img: GrayImage,
    chip_curve_pts: OpenCVIntArray,
    indirect_rotation: bool,
    thickness_majorant: int
) -> tuple[IntPtArray, FloatArray]:
    """Find the points of the chip inside contour and measure the chip thickness
    along its curve.

    thickness[i] is the measure of the chip thickness at the point
    inside_contour_pts[i]. m > n.

    Parameters
    ----------
    chip_bin_img: (h, w)-array of uint8
        Binary image containing only chip pixels.
    chip_curve_pts: (n, 1, 2)-array of int
        Points of the chip convex hull that describes the chip outside curve.
    indirect_rotation: bool
        True indicates the chip spins in the indirect direction of rotation.
    thickness_majorant: int
        Majorant of the chip thickness in pixels.

    Returns
    -------
    inside_contour_pts: (m, 2)-array of int
        Points of the chip inside contour.
    thickness: (m,)-array of float
        Thickness of the chip along its curve.
    """
    h, w = chip_bin_img.shape

    # compute the inside curve using the bisectors of the outside curve
    bisectors = compute_bisectors(chip_curve_pts, indirect_rotation)
    out_curve = chip_curve_pts.reshape(-1, 2)
    in_curve = (out_curve + thickness_majorant*bisectors).astype(np.int32)

    # remove blocks which are crossed quadrilaterals
    out_curve, in_curve = remove_crossed_quadrilaterals(out_curve, in_curve)

    # compute edges of the outside and inside curves using line rasterization
    out_curve_length = 0
    block_edges = []
    for block_idx in range(len(out_curve)-1):
        out_edge_x, out_edge_y = rasterized_line(out_curve[block_idx], out_curve[block_idx+1], h, w)
        in_edge_x, in_edge_y = rasterized_line(in_curve[block_idx], in_curve[block_idx+1], h, w)
        out_curve_length += len(out_edge_x)
        block_edges.append((out_edge_x, out_edge_y, in_edge_x, in_edge_y))

    # measure chip thickness at each point of the outside curve edges
    measure_idx = 0
    missing_mask = np.zeros(out_curve_length, dtype=bool)
    thickness = np.empty(out_curve_length, dtype=np.float64)
    inside_contour_pts = np.empty((out_curve_length, 2), dtype=np.int64)
    for out_edge_x, out_edge_y, in_edge_x, in_edge_y in block_edges:
        out_edge_length, in_edge_length = len(out_edge_x), len(in_edge_x)

        for i in range(out_edge_length):
            j = i * in_edge_length // out_edge_length

            out_x, out_y = out_edge_x[i], out_edge_y[i]
            in_x, in_y = in_edge_x[j], in_edge_y[j]
            ray_x, ray_y = rasterized_line((out_x, out_y), (in_x, in_y), h, w)

            selected_indices = np.nonzero(chip_bin_img[ray_y, ray_x])[0]
            if len(selected_indices) > 0:
                selected_x, selected_y = ray_x[selected_indices], ray_y[selected_indices]
                distances = np.linalg.norm((selected_x - out_x, selected_y - out_y), axis=0)
                innermost_idx = np.argmax(distances)
                thickness[measure_idx] = distances[innermost_idx]
                inside_contour_pts[measure_idx] = (selected_x[innermost_idx], selected_y[innermost_idx])
            else:
                missing_mask[measure_idx] = True

            measure_idx += 1

    # recreate the missing measurement by linear interpolation
    missing_idx = np.nonzero(missing_mask)[0]
    measure_idx = np.nonzero(~missing_mask)[0]
    inside_contour_pts[missing_idx, 0] = np.interp(missing_idx, measure_idx, inside_contour_pts[measure_idx, 0])
    inside_contour_pts[missing_idx, 1] = np.interp(missing_idx, measure_idx, inside_contour_pts[measure_idx, 1])
    thickness[missing_idx] = np.interp(missing_idx, measure_idx, thickness[measure_idx])

    return inside_contour_pts, thickness


NEIGHBORHOOD_MASKS = (
    cv.circle(np.zeros((9, 9), dtype=np.uint8), (4, 4), 1, 1, -1),
    cv.circle(np.zeros((9, 9), dtype=np.uint8), (4, 4), 2, 1, -1),
    cv.circle(np.zeros((9, 9), dtype=np.uint8), (4, 4), 3, 1, -1),
    cv.circle(np.zeros((9, 9), dtype=np.uint8), (4, 4), 4, 1, -1),
)


def clean_inside_contour(
    chip_bin_img: GrayImage,
    inside_contour_pts: IntPtArray,
    thickness: FloatArray,
    min_neighbors_count: int
) -> tuple[IntPtArray, FloatArray]:
    """Clean the inside contour points and thickness measures by removing some
    noisy points from the measurements.

    If a point from inside_contour_pts has less than min_neighbors_count
    neighbors, it is considered as noisy and replaced by a linear interpolation
    of its non-noisy neighbors. The more the thickness is high, the more the
    neighborhood is large.

    Parameters
    ----------
    chip_bin_img: (h, w)-array of uint8
        Binary image containing only chip pixels.
    inside_contour_pts: (m, 2)-array of int
        Points of the chip inside contour.
    thickness: (m,)-array of float
        Thickness of the chip along its curve.
    min_neighbors_count: int
        Minimum number of neighbors to consider a point as not noisy.

    Returns
    -------
    clean_inside_contour_pts: (m, 2)-array of int
        Cleaned points of the chip inside contour.
    clean_thickness: (m,)-array of float
        Cleaned thickness of the chip along its curve.
    """
    h, w = chip_bin_img.shape
    inside_contour_bin_img = np.zeros((h+8, w+8), dtype=np.uint8)
    for x, y in inside_contour_pts:
        inside_contour_bin_img[y, x] = 1

    # detect the noisy points by counting their neighbors
    max_thickness = max(thickness)
    noisy_measure_mask = np.zeros_like(thickness, dtype=bool)
    for measure_idx in range(len(thickness)):
        x, y = inside_contour_pts[measure_idx]
        t = thickness[measure_idx]
        mask_index = round(3 * t / max_thickness)
        mask = NEIGHBORHOOD_MASKS[mask_index]
        neighbor_count = np.sum(inside_contour_bin_img[y-4:y+5, x-4:x+5] * mask)
        if neighbor_count <= min_neighbors_count:
            noisy_measure_mask[measure_idx] = True

    # replace the noisy points by linear interpolations
    clean_inside_contour_pts = inside_contour_pts.copy()
    clean_thickness = thickness.copy()
    noisy_idx = np.nonzero(noisy_measure_mask)[0]
    clean_idx = np.nonzero(~noisy_measure_mask)[0]
    clean_inside_contour_pts[noisy_idx, 0] = np.interp(noisy_idx, clean_idx, inside_contour_pts[clean_idx, 0])
    clean_inside_contour_pts[noisy_idx, 1] = np.interp(noisy_idx, clean_idx, inside_contour_pts[clean_idx, 1])
    clean_thickness[noisy_idx] = np.interp(noisy_idx, clean_idx, thickness[clean_idx])

    return clean_inside_contour_pts, clean_thickness


def extract_inside_features(
    main_ft: MainFeatures,
    outside_segments: OpenCVIntArray,
    chip_binary_img: GrayImage
) -> InsideFeatures:
    inside_ft = InsideFeatures()

    inside_ft.noised_inside_contour_pts, noised_thickness = find_inside_contour(
        chip_binary_img,
        outside_segments,
        main_ft.indirect_rotation,
        thickness_majorant=125
    )
    inside_ft.inside_contour_pts, inside_ft.thickness = clean_inside_contour(
        chip_binary_img,
        inside_ft.noised_inside_contour_pts,
        noised_thickness,
        min_neighbors_count=2
    )

    return inside_ft


def render_inside_features(frame_num: int, render: ColorImage, main_ft: MainFeatures, inside_ft: InsideFeatures) -> None:
    """Draw a representation of features `main_ft` and `inside_ft` on image `render`."""
    cv.putText(render, f"frame: {frame_num}", (20, render.shape[0]-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors.WHITE)
    for x, y in inside_ft.noised_inside_contour_pts:
        render[y, x] = colors.RED
    for x, y in inside_ft.inside_contour_pts:
        render[y, x] = colors.GREEN
    # for x, y in inside_ft.chip_curve_pts.reshape(-1, 2):
    #     cv.circle(render, (x, y), 3, colors.BLUE, -1)
