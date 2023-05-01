import unittest

import numpy as np

from tensegrity.prism import Prism
from tensegrity.stiffness import rigid_body_constraint, apply_constraint, eigenmodes


class TestStiffness(unittest.TestCase):

    def test_stiffness(self):

        s = Prism(3)
        v, F, K, M = s.stiffness()
        v = np.linalg.eigvalsh(K)
        # 6 rigid body nodes
        self.assertEqual(np.count_nonzero(v < 1e-3), 6)
        # 6 rigid body nodes + 1 soft node
        self.assertEqual(np.count_nonzero(v < 1e7), 7)

    def test_rigid_body(self):

        s = Prism(3)
        # calculate stiffness
        v, F, K, M = s.stiffness()
        # get rigid body constraint
        R = rigid_body_constraint(s.nodes)
        # apply constraint
        Kr, T = apply_constraint(R, K)[:2]
        # calculate eigenvalues
        v = np.linalg.eigvalsh(Kr)
        # no rigid body nodes
        self.assertEqual(np.count_nonzero(v < 1e-3), 0)
        # 1 soft node
        self.assertEqual(np.count_nonzero(v < 1e7), 1)

    def test_eigenmodes(self):
        s = Prism(3)
        # calculate stiffness
        v, F, K, M = s.stiffness()
        # eigenmodes
        d, V = eigenmodes(K, M)
        # get rigid body constraint
        R = rigid_body_constraint(s.nodes)
        # apply constraint
        Kr, T, Mr = apply_constraint(R, K, None, M)
        # calculate eigenvalues
        v = np.linalg.eigvalsh(Kr)
        # no rigid body nodes
        self.assertEqual(np.count_nonzero(v < 1e-3), 0)
        # 1 soft node
        self.assertEqual(np.count_nonzero(v < 1e7), 1)
        # eigenmodes
        d, V = eigenmodes(Kr, Mr, T)


if __name__ == '__main__':
    unittest.main()
