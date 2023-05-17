import unittest

import numpy as np
from tnsgrt.structure import Structure


class TestStructure(unittest.TestCase):

    def test_equilibrium(self):

        nodes = np.array([[0, 0, 0], [0, 1, 0], [2, 1, 0], [2, 0, 0]]).transpose()
        members = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]).transpose()
        s = Structure(nodes, members, number_of_strings=4)

        # equilibrium
        s.equilibrium()

        # no external force
        np.testing.assert_allclose(s.member_properties['lambda_'],
                                   np.hstack((np.ones((4,)), -np.ones((2,)))))
        self.assertTrue(np.abs(np.mean(
            s.get_member_properties(s.get_members_by_tag('bar'),
                                    'lambda_')) + 1) < 1e-6)

        # equilibrium
        s.equilibrium(lambda_bar=2)

        # no external force, normalize not one
        np.testing.assert_allclose(s.member_properties['lambda_'],
                                   np.hstack((2*np.ones((4,)), -2*np.ones((2,)))))
        self.assertTrue(np.abs(np.mean(
            s.get_member_properties(s.get_members_by_tag('bar'),
                                    'lambda_')) + 2) < 1e-6)

        # external force
        f = np.array([[0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0]]).transpose()

        # equilibrium
        s.equilibrium(f)

        # no external force, normalize not one
        lmbda = np.array([0., 1., 0., 1., -1., -1.])
        np.testing.assert_allclose(s.member_properties['lambda_'], lmbda, atol=1e-6)

        # equilibrium
        s.equilibrium(f/2)

        # no external force, normalize not one
        lmbda = np.array([0., 1/2., 0., 1/2., -1/2., -1/2.])
        np.testing.assert_allclose(s.member_properties['lambda_'], lmbda, atol=1e-6)

        # equilibrium
        s.equilibrium(f/2, lambda_bar=2)

        # no external force, normalize not one
        lmbda = \
            np.array([0., 1/2., 0., 1/2., -1/2., -1/2.]) - \
            3/2*np.array([-1., -1., -1., -1., 1., 1])
        np.testing.assert_allclose(s.member_properties['lambda_'], lmbda, atol=1e-6)
        self.assertTrue(np.abs(np.mean(
            s.get_member_properties(s.get_members_by_tag('bar'),
                                    'lambda_')) + 2) < 1e-6)

    def test_prestress(self):

        nodes = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]]).transpose()
        members = np.array([[0, 1], [1, 2], [0, 2]]).transpose()
        s = Structure(nodes, members, number_of_strings=2)

        # equilibrium
        s.equilibrium()

        # no external force
        np.testing.assert_allclose(s.member_properties['lambda_'],
                                   np.hstack((2*np.ones((2,)), -np.ones((1,)))))
        self.assertTrue(np.abs(np.mean(
            s.get_member_properties(s.get_members_by_tag('bar'),
                                    'lambda_')) + 1) < 1e-6)

        # equilibrium
        s.equilibrium(lambda_bar=2)

        # no external force, normalize not one
        np.testing.assert_allclose(s.member_properties['lambda_'],
                                   np.hstack((4*np.ones((2,)), -2*np.ones((1,)))))
        self.assertTrue(np.abs(np.mean(
            s.get_member_properties(s.get_members_by_tag('bar'),
                                    'lambda_')) + 2) < 1e-6)

        # external force
        f = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]]).transpose()

        # equilibrium
        s.equilibrium(f)

        # no external force, normalize not one
        lmbda = np.array([0., 1., -1/2.])
        np.testing.assert_allclose(s.member_properties['lambda_'], lmbda, atol=1e-6)

        # equilibrium
        s.equilibrium(f/2)

        # no external force, normalize not one
        np.testing.assert_allclose(s.member_properties['lambda_'], lmbda/2, atol=1e-6)

        # equilibrium
        s.equilibrium(f, lambda_bar=2)

        # no external force, normalize not one
        lmbda = np.array([0., 1., -1/2.]) - 3/2*np.array([-2., -2., 1.])
        np.testing.assert_allclose(s.member_properties['lambda_'], lmbda, atol=1e-6)
        self.assertTrue(np.abs(np.mean(
            s.get_member_properties(
                s.get_members_by_tag('bar'), 'lambda_')) + 2) < 1e-6)
