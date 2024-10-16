from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from type_hints import GrayImage, ColorImage, IntPt, IntPtArray, IntArray, FloatPtArray, FloatArray
    from measure import ToolTipFeatures
    from features_main import MainFeatures

import numpy as np
import cv2 as cv
from skimage.draw import line

import colors


class InsideFeatures:
    __slots__ = (
        "noised_inside_contour_pts",  # type: IntPtArray
        "noised_thickness",           # type: FloatArray

        "inside_contour_pts",  # type: IntPtArray
        "thickness"            # type: FloatArray
    )


class ThicknessAnalysis:
    __slots__ = (
        "rough_thk",     # type: FloatArray
        "smoothed_thk",  # type: FloatArray

        "rough_peak_indices",  # type: IntArray
        "peak_indices",        # type: IntArray
        "valley_indices",      # type: IntArray

        "mean_peak_thickness",    # type: float
        "mean_valley_thickness",  # type: float
    )


def rasterized_line(p0: IntPt, p1: IntPt, img_height: int, img_width: int) -> tuple[IntArray, IntArray]:
    """Return the rasterization of the line segment defined by two points.

    Parameters
    ----------
    p0, p1: int couples
        Points defining the segment.
    img_height, img_width: int
        Height and width of the image.

    Returns
    -------
    raster_x, raster_y: (n,)-arrays of int
        Respectively the x and y coordinates of the image pixels crossed by the
        line segment (p0, p1). 0 <= x < img_width and 0 <= y < img_height.
    """
    line_x, line_y = line(*p0, *p1)
    inside_mask = (0 <= line_x) & (line_x < img_width) & (0 <= line_y) & (line_y < img_height)
    raster_x, raster_y = line_x[inside_mask], line_y[inside_mask]
    return raster_x, raster_y


def compute_bisectors(
    chip_curve_pts: IntPtArray,
    indirect_chip_rotation: bool
) -> FloatPtArray:
    """Return the unit vectors bisecting the edges of the chip curve.

    Vector bisectors[i] is bisecting the two adjacent edges
    (chip_curve_pts[i-1], chip_curve[i]) and (chip_curve[i], chip_curve[i+1]).
    Vector bisectors[0] is normal to the edge (chip_curve_pts[0], chip_curve_pts[1]).
    Vector bisectors[-1] is normal to the edge (chip_curve_pts[-2], chip_curve_pts[-1]).

    Parameters
    ----------
    chip_curve_pts: (n, 2)-array of int
        Points of the chip convex hull that describes the chip curve. n >= 2.
    indirect_chip_rotation: bool
        True indicates the chip spins in the indirect direction of rotation.

    Returns
    -------
    bisectors: (n, 2)-array of float
        bisector unit vectors
    """
    bisectors = np.zeros_like(chip_curve_pts, dtype=np.float32)

    u = chip_curve_pts[:-2] - chip_curve_pts[1:-1]
    v = chip_curve_pts[2:] - chip_curve_pts[1:-1]
    w = v * (np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)).reshape(-1, 1)

    # numerical instability correction if the angle between u and v is greater than pi/2
    stable = (np.sum(u*v, axis=1) > 0)
    unstable = ~stable

    bisectors[1:-1][stable] = u[stable] + w[stable]
    normals = u[unstable] - w[unstable]

    normal_first = chip_curve_pts[0] - chip_curve_pts[1]
    normal_last = chip_curve_pts[-2] - chip_curve_pts[-1]
    if indirect_chip_rotation:
        bisectors[1:-1][unstable] = np.column_stack((-normals[:, 1], normals[:, 0]))
        bisectors[0] = (-normal_first[1], normal_first[0])
        bisectors[-1] = (-normal_last[1], normal_last[0])
    else:
        bisectors[1:-1][unstable] = np.column_stack((normals[:, 1], -normals[:, 0]))
        bisectors[0] = (normal_first[1], -normal_first[0])
        bisectors[-1] = (normal_last[1], -normal_last[0])

    return bisectors / np.linalg.norm(bisectors, axis=1).reshape(-1, 1)


def remove_crossed_quadrilaterals(
    out_curve: IntPtArray,
    in_curve: IntPtArray
) -> tuple[IntPtArray, IntPtArray]:
    """Return the inside and outside curves of the chip without crossed
    quadrilaterals.

    (filt_out_curve[i], filt_in_curve[i], filt_in_curve[i+1], filt_out_curve[i+1])
    is not a crossed quadrilateral.

    Parameters
    ----------
    out_curve, in_curve: (n, 2)-arrays of int
        Respectively the outside and inside curves of the chip. n >= 2.

    Returns
    -------
    filtered_out_curve, filtered_in_curve: (m, 2)-arrays of int
        Respectively the outside and inside curves of the chip without crossed
        quadrilaterals. 2 <= m <= n.
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
    out_curve: IntPtArray,
    indirect_rotation: bool,
    thickness_majorant: float
) -> tuple[IntPtArray, FloatArray]:
    """Find the points of the chip inside contour and measure the chip thickness
    along its curve.

    thickness[i] is the measure of the chip thickness at the point
    inside_contour_pts[i]. m > n.

    Parameters
    ----------
    chip_bin_img: (h, w)-array of uint8
        Binary image containing only chip pixels.
    out_curve: (n, 2)-array of int
        Points of the chip convex hull that describes the chip outside curve.
    indirect_rotation: bool
        True indicates the chip spins in the indirect direction of rotation.
    thickness_majorant: float
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
    if len(out_curve) < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    bisectors = compute_bisectors(out_curve, indirect_rotation)
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

    # measure chip thickness at each point of the outside curve's edges, using ray castings
    pt_idx = 0
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
                thickness[pt_idx] = distances[innermost_idx]
                inside_contour_pts[pt_idx] = (selected_x[innermost_idx], selected_y[innermost_idx])
            else:
                missing_mask[pt_idx] = True

            pt_idx += 1

    # recreate the missing measurement by linear interpolation
    missing_idx = np.nonzero(missing_mask)[0]
    measure_idx = np.nonzero(~missing_mask)[0]
    if len(measure_idx) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
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
    clean_inside_contour_pts = inside_contour_pts.copy()
    clean_thickness = thickness.copy()

    h, w = chip_bin_img.shape
    inside_contour_bin_img = np.zeros((h+8, w+8), dtype=np.uint8)
    for x, y in inside_contour_pts:
        inside_contour_bin_img[y, x] = 1

    # detect the noisy points by counting their neighbors
    if len(thickness) == 0:
        return clean_inside_contour_pts, clean_thickness
    max_thickness = np.max(thickness)
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
    noisy_idx = np.nonzero(noisy_measure_mask)[0]
    clean_idx = np.nonzero(~noisy_measure_mask)[0]
    if len(clean_idx) == 0:
        return clean_inside_contour_pts, clean_thickness
    clean_inside_contour_pts[noisy_idx, 0] = np.interp(noisy_idx, clean_idx, inside_contour_pts[clean_idx, 0])
    clean_inside_contour_pts[noisy_idx, 1] = np.interp(noisy_idx, clean_idx, inside_contour_pts[clean_idx, 1])
    clean_thickness[noisy_idx] = np.interp(noisy_idx, clean_idx, thickness[clean_idx])

    return clean_inside_contour_pts, clean_thickness


def extract_inside_features(
    main_ft: MainFeatures,
    out_curve: IntPtArray,
    chip_binary_img: GrayImage,
    tool_penetration: float
) -> InsideFeatures:
    """Extract the features concerning the inside contour of the chip.

    Parameters
    ----------
    main_ft: MainFeatures
        Main features of the preprocessed machining image.
    out_curve: (n, 2)-array of int
        Points of the chip convex hull that describes the chip outside curve.
    chip_binary_img: (h, w)-array of uint8
        Machining binary image containing only the chip pixels.
    tool_penetration: float
        Tool penetration length into the part being machined.

    Returns
    -------
    inside_ft: InsideFeatures
        Structure containing the extracted features:
        - the points of the chip inside contour. It is a non-convex contour made
        of peaks and valleys.
        - the thickness of the chip along its curve.
    """
    inside_ft = InsideFeatures()

    inside_ft.noised_inside_contour_pts, inside_ft.noised_thickness = find_inside_contour(
        chip_binary_img,
        out_curve,
        main_ft.indirect_rotation,
        thickness_majorant=2*tool_penetration
    )
    inside_ft.inside_contour_pts, inside_ft.thickness = clean_inside_contour(
        chip_binary_img,
        inside_ft.noised_inside_contour_pts,
        inside_ft.noised_thickness,
        min_neighbors_count=2
    )

    return inside_ft


def render_inside_features(
    frame_num: int,
    render: ColorImage,
    inside_ft: InsideFeatures,
    thk_an: ThicknessAnalysis
) -> None:
    assert render.ndim == 3 and render.shape[2] == 3
    # draw the detected inside contour of the chip
    for i in range(len(inside_ft.noised_inside_contour_pts)-1):
        pt0 = inside_ft.noised_inside_contour_pts[i]
        pt1 = inside_ft.noised_inside_contour_pts[i+1]
        cv.line(render, pt0, pt1, colors.RED, thickness=1)

    # draw the denoised inside contour of the chip
    for i in range(len(inside_ft.inside_contour_pts)-1):
        pt0 = inside_ft.inside_contour_pts[i]
        pt1 = inside_ft.inside_contour_pts[i+1]
        cv.line(render, pt0, pt1, colors.GREEN, thickness=1)

    # draw the detected valley and peaks
    for pt in inside_ft.inside_contour_pts[thk_an.peak_indices]:
        cv.circle(render, pt, 4, colors.YELLOW, thickness=-1)
    for pt in inside_ft.inside_contour_pts[thk_an.valley_indices]:
        cv.circle(render, pt, 4, colors.YELLOW, thickness=-1)

    # write the frame number
    cv.putText(render, f"frame: {frame_num}", (20, render.shape[0]-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors.WHITE)
