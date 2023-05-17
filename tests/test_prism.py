import unittest

import itertools
from parameterized import parameterized
import numpy as np

from tnsgrt.prism import Prism


class TestSnelson(unittest.TestCase):

    @parameterized.expand(itertools.product([3, 4, 6, 12], ['lp', 'analytic']))
    def test_regular_prism(self, p, method='analytic'):
        s = Prism(p, equilibrium_method=method)
        self.assertEqual(s.get_number_of_members(), 4*p)
        self.assertEqual(s.get_number_of_nodes(), 2*p)
        self.assertEqual(set(s.member_tags.keys()),
                         {'bar', 'string', 'vertical', 'top', 'bottom'})

        rho = 1
        alpha = np.pi/2-np.pi/p
        lambda_ = np.hstack((
            rho * np.cos(np.pi/p) / np.cos(alpha - np.pi/p) * np.ones((p,)),
            (1/rho) * np.cos(np.pi / p) / np.cos(alpha - np.pi/p) * np.ones((p,)),
            np.ones((p,)),
            -np.ones((p,))
        ))
        np.testing.assert_allclose(s.member_properties['lambda_'], lambda_)

    @parameterized.expand(itertools.product([3, 4, 6, 12],
                                            [0, .25, .5, 1],
                                            ['lp', 'analytic'], [1, 0.8, 1.2]))
    def test_diagonal_prism(self, p, lb=0, equilibrium_method='analytic', rho=1):
        alpha = (1-lb) * (np.pi/2 - np.pi/p) + lb * np.pi/2
        s = Prism(p, top_radius=rho, bottom_radius=1, alpha=alpha, diagonal=True,
                  equilibrium_method=equilibrium_method)
        self.assertEqual(s.get_number_of_members(), 5 * p)
        self.assertEqual(s.get_number_of_nodes(), 2 * p)
        self.assertEqual(set(s.member_tags.keys()),
                         {'bar', 'string', 'vertical', 'top', 'bottom', 'diagonal'})

        lambda_ = np.hstack((
            rho * np.cos(np.pi / p) / np.cos(alpha - np.pi / p) * np.ones((p,)),
            (1 / rho) * np.cos(np.pi / p) / np.cos(alpha - np.pi / p) * np.ones((p,)),
            2 * np.cos(alpha) * np.cos(np.pi / p)
            / np.cos(alpha - np.pi / p) * np.ones((p,)),
            -np.cos(alpha + np.pi / p) / np.cos(alpha - np.pi / p) * np.ones((p,)),
            -np.ones((p,))
        ))
        np.testing.assert_allclose(
            s.member_properties['lambda_'], lambda_, atol=1e-6, rtol=0)


if __name__ == '__main__':
    unittest.main()
