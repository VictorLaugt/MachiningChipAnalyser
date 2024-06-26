from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, TypeVar
    PointArray = TypeVar('PointArray')  # ~ (n, 1, 2) dtype=int32
    Point = TypeVar('Point')

import numpy as np

def inside_polygon(points: PointArray, polygon_vertices: Sequence[Point]) -> np.ndarray[bool]:
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)

    with np.errstate(divide='ignore'):
        xa, ya = polygon_vertices[-1]
        for xb, yb in polygon_vertices:
            cond1 = (ya > y) ^ (yb > y)
            cond2 = x < (xb - xa) * (y - ya) / (yb - ya) + xa
            inside ^= cond1 & cond2
            xa, ya = xb, yb

    for xv, yv in polygon_vertices:
        inside &= (x != xv) | (y != yv)

    return inside


def tri_to_rect_mapping(pts: PointArray, a: Point, b: Point, c: Point) -> PointArray:
    p = pts.reshape(-1, 2)
    a, b, c = np.array(a), np.array(b), np.array(c)

    ab = b - a
    base_u = ab / np.linalg.norm(ab)
    base_v = np.array((-base_u[1], base_u[0]))
    transition_matrix = np.column_stack((base_u, base_v))

    b_uv = (b - a) @ transition_matrix
    c_uv = (c - a) @ transition_matrix
    p_uv = (p - a) @ transition_matrix

    ac_uv = c_uv
    bc_uv = c_uv - b_uv
    ac_projs_u = ac_uv[0] / ac_uv[1] * p_uv[:, 1]
    bc_projs_u = bc_uv[0] / bc_uv[1] * p_uv[:, 1] + b_uv[0]

    mapped_p_u = np.abs(b_uv[0] / (bc_projs_u - ac_projs_u)) * (p_uv[:, 0] - ac_projs_u)
    mapped_p = a + mapped_p_u.reshape(-1, 1)*base_u + p_uv[:, 1].reshape(-1, 1)*base_v
    return mapped_p.reshape(pts.shape)


def tri_to_rect_unmapping(pts: PointArray, a: Point, b: Point, c: Point) -> PointArray:
    p = pts.reshape(-1, 2)
    a, b, c = np.array(a), np.array(b), np.array(c)

    ab = b - a
    base_u = ab / np.linalg.norm(ab)
    base_v = np.array((-base_u[1], base_u[0]))
    transition_matrix = np.column_stack((base_u, base_v))

    b_uv = (b - a) @ transition_matrix
    c_uv = (c - a) @ transition_matrix
    p_uv = (p - a) @ transition_matrix

    ac_uv = c_uv
    bc_uv = c_uv - b_uv
    ac_projs_u = ac_uv[0] / ac_uv[1] * p_uv[:, 1]
    bc_projs_u = bc_uv[0] / bc_uv[1] * p_uv[:, 1] + b_uv[0]

    unmapped_p_u = np.abs((bc_projs_u - ac_projs_u) / b_uv[0]) * p_uv[:, 0] + ac_projs_u
    unmapped_p = a + unmapped_p_u.reshape(-1, 1)*base_u + p_uv[:, 1].reshape(-1, 1)*base_v
    return unmapped_p.reshape(pts.shape)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a = np.array([10, 15])
    b = np.array([50, 10])
    c = np.array([40, 40])

    x_min, x_max = min(a[0], b[0], c[0]), max(a[0], b[0], c[0])
    y_min, y_max = min(a[1], b[1], c[1]), max(a[1], b[1], c[1])

    nb_pts = 1000
    pts = np.column_stack((np.random.uniform(x_min, x_max, nb_pts), np.random.uniform(y_min, y_max, nb_pts)))
    pts = pts[inside_polygon(pts, [a, b, c])]
    mapped_pts = tri_to_rect_mapping(pts, a, b, c)
    unmapped_pts = tri_to_rect_unmapping(mapped_pts, a, b, c)

    x_min, x_max = min(np.min(mapped_pts[:, 0]), x_min), max(np.max(mapped_pts[:, 0]), x_max)
    y_min, y_max = min(np.min(mapped_pts[:, 1]), y_min), max(np.max(mapped_pts[:, 1]), y_max)

    fig, axis = plt.subplots(1, 3, figsize=(10, 5))
    axis[0].plot(*zip(a, b, c, a), 'ro-')
    axis[0].scatter(pts[:, 0], pts[:, 1])
    axis[0].set_xlim(x_min, x_max)
    axis[0].set_ylim(y_min, y_max)

    axis[1].scatter(mapped_pts[:, 0], mapped_pts[:, 1])
    axis[1].set_xlim(x_min, x_max)
    axis[1].set_ylim(y_min, y_max)

    axis[2].scatter(unmapped_pts[:, 0], unmapped_pts[:, 1])
    axis[2].set_xlim(x_min, x_max)
    axis[2].set_ylim(y_min, y_max)

    plt.show()

