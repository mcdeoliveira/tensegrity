import unittest

import numpy as np

from tensegrity.snelson import Snelson


class TestSnelson(unittest.TestCase):

    def test_constructor(self):

        # empty structure
        s = Snelson()
        self.assertEqual(s.label, None)
        self.assertEqual(s.nodes.dtype, np.float_)
        self.assertEqual(s.members.dtype, np.uint64)
        # np.testing.assert_array_equal(s.nodes, nodes)
        # np.testing.assert_array_equal(s.members, members)
        # np.testing.assert_array_equal(s.member_tags['bar'], np.arange(1, 3, dtype=np.uint64))
        # np.testing.assert_array_equal(s.member_tags['string'], np.arange(0, 1, dtype=np.uint64))
        # np.testing.assert_array_equal(s.member_tags['vertical'], np.arange(2, 3, dtype=np.uint64))
        # self.assertEqual(s.get_number_of_nodes(), 3)
        # self.assertEqual(s.get_number_of_members(), 3)
        # self.assertEqual(s.get_member_tags(0), ['string'])
        # self.assertEqual(s.get_member_tags(1), ['bar'])
        # self.assertEqual(s.get_member_tags(2), ['bar', 'vertical'])
        # self.assertTrue(s.has_member_tag(2, 'bar'))
        # self.assertTrue(s.has_member_tag(2, 'vertical'))
        # self.assertFalse(s.has_member_tag(1, 'vertical'))
        # np.testing.assert_array_equal(s.get_members_by_tags(['bar']), [1, 2])
        # np.testing.assert_array_equal(s.get_members_by_tags(['bar', 'vertical']), [2])
        # self.assertEqual(len(s.member_properties), 3)
        # self.assertEqual(s.member_properties.loc[1, 'linewidth'], 2)
        # self.assertEqual(s.member_properties.loc[2, 'linewidth'], 1001)
        # self.assertEqual(s.get_member_properties([1, 2], ['volume', 'mass']).to_dict('index'),
        #                  {1: {'volume': 0., 'mass': 1.}, 2: {'volume': 2., 'mass': 1.}})
        # self.assertEqual(s.get_member_properties(2, ['volume', 'mass']).to_dict(), {'volume': 2., 'mass': 1.})


if __name__ == '__main__':
    unittest.main()
