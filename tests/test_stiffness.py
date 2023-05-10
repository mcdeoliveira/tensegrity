import unittest
from typing import Optional

import numpy as np
import scipy

from tensegrity.prism import Prism
from tensegrity.stiffness import Stiffness, NodeConstraint


class TestStiffness(unittest.TestCase):

    def test_stiffness(self):
        s = Prism(3)
        S, F, v = s.stiffness()
        d = S.eigs()[0]
        # 6 rigid body nodes
        self.assertEqual(np.count_nonzero(d < 1e-3), 6)
        # 6 rigid body nodes + 1 soft node
        self.assertEqual(np.count_nonzero(d < 1e7), 7)

    def test_rigid_body(self):
        s = Prism(3)
        # calculate stiffness
        S, F, v = s.stiffness(apply_rigid_body_constraint=True)
        # calculate eigenvalues
        d = S.eigs(k=6)[0]
        # no rigid body nodes
        self.assertEqual(np.count_nonzero(d < 1e-3), 0)
        # 1 soft node
        self.assertEqual(np.count_nonzero(d < 1e7), 1)

    def test_eigenmodes(self):
        s = Prism(3)
        # calculate stiffness
        S, F, v = s.stiffness(apply_rigid_body_constraint=True)
        # eigenmodes
        d = S.modes(k=6)[0]
        # no rigid body nodes
        self.assertEqual(np.count_nonzero(d < 1e-3), 0)
        # 1 soft node
        self.assertEqual(np.count_nonzero(d < 1e2), 1)

    def test_node_constraint(self):

        c = NodeConstraint()
        self.assertEqual(c.dof, 0)
        np.testing.assert_allclose(c.normal, np.eye(3))
        self.assertEqual(c.basis, None)

        c = NodeConstraint(np.array([[1, 0, 0]]))
        self.assertEqual(c.dof, 2)
        np.testing.assert_allclose(c.normal, np.array([[1, 0, 0]]))
        np.testing.assert_allclose(c.basis, [[0, 0], [1, 0], [0, 1]])

        c = NodeConstraint(np.array([[1, 0, 0], [0, 1, 0]]))
        self.assertEqual(c.dof, 1)
        np.testing.assert_allclose(c.normal, np.array([[1, 0, 0], [0, 1, 0]]))
        np.testing.assert_allclose(c.basis, [[0], [0], [1]])

        c = NodeConstraint(np.array([[1, 1, 0]]))
        self.assertEqual(c.dof, 2)
        np.testing.assert_allclose(c.normal, np.array([[-np.sqrt(2) / 2, -np.sqrt(2) / 2, 0]]))
        np.testing.assert_allclose(c.basis, [[-np.sqrt(2) / 2, 0], [np.sqrt(2) / 2, 0], [0, 1]])

        nodes = np.array([[1, 0], [0, 1], [0, 0]])
        constraints = [None]*2
        constraints[0] = NodeConstraint(np.array(([[1, 1, 0]])))
        constraints[1] = NodeConstraint()

        R, T = NodeConstraint.node_constraint(nodes, constraints, storage='full')
        self.assertEqual(R.shape, (4, 6))
        self.assertEqual(T.shape, (6, 2))

        Rs, Ts = NodeConstraint.node_constraint(nodes, constraints)
        self.assertEqual(Rs.shape, (4, 6))
        self.assertEqual(Ts.shape, (6, 2))

        np.testing.assert_equal(R, Rs.toarray())
        np.testing.assert_equal(T, Ts.toarray())

        self.assertTrue(np.linalg.norm((Rs @ Ts).toarray()) < 1e-6)
        self.assertTrue(np.linalg.norm(Rs @ Rs.transpose() - np.eye(4)) < 1e-6)
        self.assertTrue(np.linalg.norm(Ts.transpose() @ Ts - np.eye(2)) < 1e-6)

        nodes = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
        constraints: list[Optional[NodeConstraint]] = [None]*3
        constraints[0] = NodeConstraint(np.array(([[1, 1, 0]])))
        constraints[2] = NodeConstraint()

        R, T = NodeConstraint.node_constraint(nodes, constraints, storage='full')
        self.assertEqual(R.shape, (4, 9))
        self.assertEqual(T.shape, (9, 5))

        Rs, Ts = NodeConstraint.node_constraint(nodes, constraints)
        self.assertEqual(Rs.shape, (4, 9))
        self.assertEqual(Ts.shape, (9, 5))

        np.testing.assert_equal(R, Rs.toarray())
        np.testing.assert_equal(T, Ts.toarray())

        self.assertTrue(np.linalg.norm((Rs @ Ts).toarray()) < 1e-6)
        self.assertTrue(np.linalg.norm(Rs @ Rs.transpose() - np.eye(4)) < 1e-6)
        self.assertTrue(np.linalg.norm(Ts.transpose() @ Ts - np.eye(5)) < 1e-6)

        nodes = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 1], [0, 0, 2, 1, 3]])
        constraints: list[Optional[NodeConstraint]] = [None]*5
        constraints[0] = NodeConstraint(np.array(([[1, 1, 0]])))
        constraints[2] = NodeConstraint()

        R, T = NodeConstraint.node_constraint(nodes, constraints, storage='full')
        self.assertEqual(R.shape, (4, 15))
        self.assertEqual(T.shape, (15, 11))

        Rs, Ts = NodeConstraint.node_constraint(nodes, constraints)
        self.assertEqual(Rs.shape, (4, 15))
        self.assertEqual(Ts.shape, (15, 11))

        np.testing.assert_equal(R, Rs.toarray())
        np.testing.assert_equal(T, Ts.toarray())

        self.assertTrue(np.linalg.norm((Rs @ Ts).toarray()) < 1e-6)
        self.assertTrue(np.linalg.norm(Rs @ Rs.transpose() - np.eye(4)) < 1e-6)
        self.assertTrue(np.linalg.norm(Ts.transpose() @ Ts - np.eye(11)) < 1e-6)


if __name__ == '__main__':
    unittest.main()
