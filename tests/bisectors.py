import unittest

import numpy as np
import matplotlib.pyplot as plt

from shape_detection.chip_inside_contour import compute_bisectors


def angle_between_vectors(u, v):
    dot_products = np.sum(u*v, axis=1)
    norm_products = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    return np.arccos(dot_products / norm_products)


def check_middle_angles_bisection(pts, bisectors):
    u = pts[:-2] - pts[1:-1]
    v = pts[2:] - pts[1:-1]

    ub_angles = angle_between_vectors(u, bisectors[1:-1])
    vb_angles = angle_between_vectors(v, bisectors[1:-1])

    success = np.isclose(ub_angles, vb_angles, atol=1e-5)
    if success.all():
        return True
    else:
        fail = ~success
        print(f"ub fails = {ub_angles[fail]}")
        print(f"vb fails = {vb_angles[fail]}")
        print(f"errors = {ub_angles[fail] - vb_angles[fail]}")

    return np.isclose(ub_angles, vb_angles, atol=1e-5).all()


def check_bound_angles_are_90(pts, bisectors):
    first_bisector, last_bisector = bisectors[0], bisectors[-1]
    first_edge, last_edge = (pts[1] - pts[0]), (pts[-1] - pts[-2])

    first_bound_angle = np.arccos(np.dot(first_bisector, first_edge) / np.linalg.norm(first_edge))
    last_bound_angle = np.arccos(np.dot(last_bisector, last_edge) / np.linalg.norm(last_edge))

    return (
        np.isclose(first_bound_angle, np.pi/2) and
        np.isclose(last_bound_angle, np.pi/2)
    )


def are_unitary(vectors):
    return np.isclose(np.linalg.norm(vectors, axis=1), 1.0).all()


class TestBisectorAlgorithm(unittest.TestCase):
    def _test_on_points(self, pts, direct_rotation=True, interactive=False):
        bisectors = compute_bisectors(pts.reshape(-1, 1, 2), not direct_rotation)

        if interactive:
            _fig, ax = plt.subplots()
            ax.set_aspect('equal', adjustable='box')
            ax.grid()
            ax.plot(pts[:, 0], pts[:, 1], '-x')
            ax.quiver(pts[:, 0], pts[:, 1], bisectors[:, 0], bisectors[:, 1])
            plt.show()

        self.assertTrue(are_unitary(bisectors))
        self.assertTrue(check_middle_angles_bisection(pts, bisectors))
        self.assertTrue(check_bound_angles_are_90(pts, bisectors))

    def test_triangle(self):
        pts = np.array([[0., 0.], [1., 1.], [2., 0.]])
        self._test_on_points(pts, direct_rotation=True)
        self._test_on_points(pts, direct_rotation=False)

    def test_curve_evenly_spaced(self):
        theta = np.linspace(0, 2*np.pi/3, 10)
        pts = np.column_stack((np.cos(theta), np.sin(theta)))
        self._test_on_points(pts, direct_rotation=True)
        self._test_on_points(pts, direct_rotation=False)

    def test_curve_randomly_spaced(self):
        theta = np.random.rand(10)
        theta.sort()
        theta *= 2*np.pi/3
        pts = np.column_stack((np.cos(theta), np.sin(theta)))
        self._test_on_points(pts, direct_rotation=True)
        self._test_on_points(pts, direct_rotation=False)

    def test_random_points(self):
        pts = np.unique(np.random.randint(0, 500, (100, 2)), axis=0)
        self._test_on_points(pts, direct_rotation=True)

    def test_many_random_points(self):
        pts = np.unique(np.random.rand(100000, 2), axis=0) * 1e3
        self._test_on_points(pts, direct_rotation=True)


if __name__ == "__main__":
    unittest.main()
