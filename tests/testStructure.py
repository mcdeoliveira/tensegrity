import unittest

import numpy as np
import pandas as pd

from tensegrity.structure import Structure


class TestStructure(unittest.TestCase):

    def test_constructor(self):

        # empty structure
        s = Structure()
        self.assertEqual(s.label, None)
        np.testing.assert_array_equal(s.nodes, np.zeros((3, 0), np.float_))
        np.testing.assert_array_equal(s.members, np.zeros((2, 0), np.uint64))
        np.testing.assert_array_equal(s.member_tags['bar'], np.zeros((0,), np.uint64))
        np.testing.assert_array_equal(s.member_tags['string'], np.zeros((0,), np.uint64))
        self.assertEqual(s.get_number_of_nodes(), 0)
        self.assertEqual(s.get_number_of_members(), 0)
        self.assertEqual(len(s.member_properties), 0)

        nodes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        s = Structure(nodes, label='label')
        self.assertEqual(s.label, 'label')
        self.assertEqual(s.nodes.dtype, np.float_)
        self.assertEqual(s.members.dtype, np.uint64)
        np.testing.assert_array_equal(s.nodes, nodes)
        np.testing.assert_array_equal(s.members, np.zeros((2, 0), np.uint64))
        np.testing.assert_array_equal(s.member_tags['bar'], np.zeros((0,), np.uint64))
        np.testing.assert_array_equal(s.member_tags['string'], np.zeros((0,), np.uint64))
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.get_number_of_members(), 0)
        self.assertEqual(len(s.member_properties), 0)

        members = np.array([[0, 1, 2], [1, 2, 0]])
        s = Structure(nodes, members, label='label')
        self.assertEqual(s.label, 'label')
        self.assertEqual(s.nodes.dtype, np.float_)
        self.assertEqual(s.members.dtype, np.uint64)
        np.testing.assert_array_equal(s.nodes, nodes)
        np.testing.assert_array_equal(s.members, members)
        np.testing.assert_array_equal(s.member_tags['bar'], np.arange(0, 3, dtype=np.uint64))
        np.testing.assert_array_equal(s.member_tags['string'], np.zeros((0,), np.uint64))
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.get_number_of_members(), 3)
        self.assertEqual(len(s.member_properties), 3)

        s = Structure(nodes, members, number_of_strings=1)
        self.assertEqual(s.label, None)
        self.assertEqual(s.nodes.dtype, np.float_)
        self.assertEqual(s.members.dtype, np.uint64)
        np.testing.assert_array_equal(s.nodes, nodes)
        np.testing.assert_array_equal(s.members, members)
        np.testing.assert_array_equal(s.member_tags['bar'], np.arange(1, 3, dtype=np.uint64))
        np.testing.assert_array_equal(s.member_tags['string'], np.arange(0, 1, dtype=np.uint64))
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.get_number_of_members(), 3)
        self.assertEqual(len(s.member_properties), 3)

        s = Structure(nodes, members, number_of_strings=2)
        self.assertEqual(s.label, None)
        self.assertEqual(s.nodes.dtype, np.float_)
        self.assertEqual(s.members.dtype, np.uint64)
        np.testing.assert_array_equal(s.nodes, nodes)
        np.testing.assert_array_equal(s.members, members)
        np.testing.assert_array_equal(s.member_tags['bar'], np.arange(2, 3, dtype=np.uint64))
        np.testing.assert_array_equal(s.member_tags['string'], np.arange(0, 2, dtype=np.uint64))
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.get_number_of_members(), 3)
        self.assertEqual(len(s.member_properties), 3)

        Structure.member_defaults['vertical'] = {'linewidth': 1001, 'volume': 2}
        s = Structure(nodes, members,
                      member_tags={
                          'bar': np.arange(1, 3, dtype=np.uint64),
                          'string': np.arange(0, 1, dtype=np.uint64),
                          'vertical': np.arange(2, 3, dtype=np.uint64)
                      })
        self.assertEqual(s.label, None)
        self.assertEqual(s.nodes.dtype, np.float_)
        self.assertEqual(s.members.dtype, np.uint64)
        np.testing.assert_array_equal(s.nodes, nodes)
        np.testing.assert_array_equal(s.members, members)
        np.testing.assert_array_equal(s.member_tags['bar'], np.arange(1, 3, dtype=np.uint64))
        np.testing.assert_array_equal(s.member_tags['string'], np.arange(0, 1, dtype=np.uint64))
        np.testing.assert_array_equal(s.member_tags['vertical'], np.arange(2, 3, dtype=np.uint64))
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.get_number_of_members(), 3)
        self.assertEqual(s.get_member_tags(0), ['string'])
        self.assertEqual(s.get_member_tags(1), ['bar'])
        self.assertEqual(s.get_member_tags(2), ['bar', 'vertical'])
        self.assertTrue(s.has_member_tag(2, 'bar'))
        self.assertTrue(s.has_member_tag(2, 'vertical'))
        self.assertFalse(s.has_member_tag(1, 'vertical'))
        np.testing.assert_array_equal(s.get_elements_by_tags(['bar']), [1, 2])
        np.testing.assert_array_equal(s.get_elements_by_tags(['bar', 'vertical']), [2])
        self.assertEqual(len(s.member_properties), 3)
        self.assertEqual(s.member_properties.loc[1, 'linewidth'], 2)
        self.assertEqual(s.member_properties.loc[2, 'linewidth'], 1001)
        self.assertEqual(s.get_member_properties([1, 2], ['volume', 'mass']),
                         {1: {'volume': 0., 'mass': 1.}, 2: {'volume': 2., 'mass': 1.}})
        self.assertEqual(s.get_member_properties(2, ['volume', 'mass']), {'volume': 2., 'mass': 1.})


if __name__ == '__main__':
    unittest.main()
