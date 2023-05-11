import unittest

import numpy as np

from tnsgrt.utils import rotation_2d, rotation_3d


class TestUtils(unittest.TestCase):

    def test_rotation_2d(self):

        v = np.array([1, 0])
        rotation = rotation_2d(0)
        np.testing.assert_array_equal(rotation @ v, v)

        rotation = rotation_2d(np.pi/2)
        np.testing.assert_almost_equal(rotation @ v, np.array([0, 1]))

        rotation = rotation_2d(np.pi)
        np.testing.assert_almost_equal(rotation @ v, np.array([-1, 0]))

        rotation = rotation_2d(np.pi/4)
        np.testing.assert_almost_equal(rotation @ v, np.array([1, 1])/np.sqrt(2))

    def test_rotation_3d(self):

        v = np.array([1, 0, 0])
        z = np.array([0, 0, 1])
        rotation = rotation_3d(0 * z)
        np.testing.assert_array_equal(rotation @ v, v)

        rotation = rotation_3d(np.pi/2 * z)
        np.testing.assert_almost_equal(rotation @ v, np.array([0, 1, 0]))

        rotation = rotation_3d(np.pi * z)
        np.testing.assert_almost_equal(rotation @ v, np.array([-1, 0, 0]))

        rotation = rotation_3d(np.pi/4 * z)
        np.testing.assert_almost_equal(rotation @ v, np.array([1, 1, 0])/np.sqrt(2))

        z = np.array([0, -1, 0])
        rotation = rotation_3d(0 * z)
        np.testing.assert_array_equal(rotation @ v, v)

        rotation = rotation_3d(np.pi/2 * z)
        np.testing.assert_almost_equal(rotation @ v, np.array([0, 0, 1]))

        rotation = rotation_3d(np.pi * z)
        np.testing.assert_almost_equal(rotation @ v, np.array([-1, 0, 0]))

        rotation = rotation_3d(np.pi/4 * z)
        np.testing.assert_almost_equal(rotation @ v, np.array([1, 0, 1])/np.sqrt(2))
