import numpy as np
import matplotlib.pyplot as plt

from shape_detection.chip_inside_contour3 import compute_bisectors


def _angle_between_vectors(u, v):
    dot_products = np.sum(u*v, axis=1)
    norm_products = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    return np.arccos(dot_products / norm_products)


def _assert_middle_angles(pts, bisectors):
    u = pts[:-2] - pts[1:-1]
    v = pts[2:] - pts[1:-1]

    half_angles = _angle_between_vectors(u, v) / 2
    ub_angles = _angle_between_vectors(u, bisectors[1:-1])
    vb_angles = _angle_between_vectors(v, bisectors[1:-1])

    assert np.isclose(ub_angles, half_angles).all()
    assert np.isclose(vb_angles, half_angles).all()


def _assert_bound_angles(pts, bisectors):
    first_bisector, last_bisector = bisectors[0], bisectors[-1]
    first_edge, last_edge = pts[1]-pts[0], pts[-1]-pts[-2]

    first_bound_angle = np.arccos(np.dot(first_bisector, first_edge) / np.linalg.norm(first_edge))
    last_bound_angle = np.arccos(np.dot(last_bisector, last_edge)) / np.linalg.norm(last_edge)

    print(f"{first_bound_angle=}")
    print(f"{last_bound_angle=}")

    assert np.isclose(first_bound_angle, np.pi/2)
    assert np.isclose(last_bound_angle, np.pi/2)


def _assert_unit_vectors(vectors):
    assert np.isclose(np.linalg.norm(vectors, axis=1), 1.0).all()


def _test_on_points(pts, direct_rotation=True):
    bisectors = compute_bisectors(pts.reshape(-1, 1, 2), not direct_rotation)
    _fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.grid()
    ax.plot(pts[:, 0], pts[:, 1], '-x')
    ax.quiver(pts[:, 0], pts[:, 1], bisectors[:, 0], bisectors[:, 1])
    plt.show()

    _assert_unit_vectors(bisectors)
    _assert_middle_angles(pts, bisectors)
    _assert_bound_angles(pts, bisectors)


def test_triangle():
    pts = np.array([[.0, .0], [.5, .5], [.0, .7]])
    _test_on_points(pts)


def test__curve_evenly_spaced():
    theta = np.linspace(0, 2*np.pi/3, 10)
    pts = np.column_stack((np.cos(theta), np.sin(theta)))
    _test_on_points(pts)


def test__curve_randomly_spaced():
    theta = np.random.rand(10)
    theta.sort()
    theta *= 2*np.pi/3
    pts = np.column_stack((np.cos(theta), np.sin(theta)))
    _test_on_points(pts)

if __name__ == "__main__":
    test_triangle()
    # test__curve_evenly_spaced()
    # test__curve_randomly_spaced()
