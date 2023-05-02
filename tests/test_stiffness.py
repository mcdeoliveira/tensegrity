import unittest

import numpy as np

from tensegrity.prism import Prism
from tensegrity.stiffness import Stiffness


class TestStiffness(unittest.TestCase):

    def test_stiffness(self):

        s = Prism(3)
        v, F, K, M = s.stiffness()
        S = Stiffness(K, M=M)
        d = S.eigs()[0]
        # 6 rigid body nodes
        self.assertEqual(np.count_nonzero(d < 1e-3), 6)
        # 6 rigid body nodes + 1 soft node
        self.assertEqual(np.count_nonzero(d < 1e7), 7)

    def test_rigid_body(self):

        s = Prism(3)
        # calculate stiffness
        v, F, K, M = s.stiffness()
        S = Stiffness(K, M=M)
        # get rigid body constraint
        R = Stiffness.rigid_body_constraint(s.nodes)
        # apply constraint
        S.apply_constraint(R)
        # calculate eigenvalues
        d = S.eigs()[0]
        # no rigid body nodes
        self.assertEqual(np.count_nonzero(d < 1e-3), 0)
        # 1 soft node
        self.assertEqual(np.count_nonzero(d < 1e7), 1)

    def test_eigenmodes(self):
        s = Prism(3)
        # calculate stiffness
        v, F, K, M = s.stiffness()
        S = Stiffness(K, M=M)
        # get rigid body constraint
        R = Stiffness.rigid_body_constraint(s.nodes)
        # apply constraint
        S.apply_constraint(R)
        # eigenmodes
        d = S.eigenmodes()[0]
        # no rigid body nodes
        self.assertEqual(np.count_nonzero(d < 1e-3), 0)
        # 1 soft node
        self.assertEqual(np.count_nonzero(d < 1e4), 1)


if __name__ == '__main__':
    unittest.main()
