import unittest

import itertools
from parameterized import parameterized

from tnsgrt.michell import Michell


class TestMichell(unittest.TestCase):

    @parameterized.expand(itertools.product([4, 6, 12], [3, 4, 10, 12], [0.5, 1, 1.5]))
    def test_michell(self, p, q, radius):
        s = Michell(p, q=q, radius=radius)
        # print(f'p = {p}, q = {q}')
        self.assertEqual(s.get_number_of_nodes(), p*q+1)
        # self.assertEqual(s.get_number_of_members(), p*(q+1)+2*p)
        # self.assertEqual(set(s.member_tags.keys()),
        # {'bar', 'string', 'vertical', 'top', 'bottom'})
        #
        # rho = 1
        # alpha = np.pi/2-np.pi/p
        # lambda_ = np.hstack((
        #     rho * np.cos(np.pi/p) / np.cos(alpha - np.pi/p) * np.ones((p,)),
        #     (1/rho) * np.cos(np.pi / p) / np.cos(alpha - np.pi/p) * np.ones((p,)),
        #     np.ones((p,)),
        #     -np.ones((p,))
        # ))
        # np.testing.assert_allclose(s.member_properties['lambda_'], lambda_)


if __name__ == '__main__':
    unittest.main()
