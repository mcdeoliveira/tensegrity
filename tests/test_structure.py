import unittest

import numpy as np
import pandas as pd
import scipy

from tnsgrt import utils, structure
from tnsgrt.stiffness import NodeConstraint
from tnsgrt.structure import Structure


class TestStructure(unittest.TestCase):

    def test_constructor(self):

        # empty structure
        s = Structure()
        self.assertEqual(s.label, None)
        np.testing.assert_array_equal(s.nodes, np.zeros((3, 0), np.float_))
        self.assertEqual(len(s.node_properties), 0)
        self.assertEqual(s.get_number_of_nodes(), 0)
        self.assertEqual(s.node_tags, {})
        self.assertEqual(len(s.member_properties), 0)
        self.assertEqual(s.get_number_of_members(), 0)
        np.testing.assert_array_equal(s.members, np.zeros((2, 0), np.int64))
        np.testing.assert_array_equal(s.member_tags['bar'], np.zeros((0,), np.int64))
        np.testing.assert_array_equal(s.member_tags['string'], np.zeros((0,), np.int64))

        nodes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        s = Structure(nodes, label='label')
        self.assertEqual(s.label, 'label')
        self.assertEqual(s.nodes.dtype, np.float_)
        self.assertEqual(s.members.dtype, np.int64)
        np.testing.assert_array_equal(s.nodes, nodes)
        self.assertEqual(len(s.node_properties), 3)
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.node_tags, {})
        np.testing.assert_array_equal(s.members, np.zeros((2, 0), np.int64))
        np.testing.assert_array_equal(s.member_tags['bar'], np.zeros((0,), np.int64))
        np.testing.assert_array_equal(s.member_tags['string'], np.zeros((0,), np.int64))
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.get_number_of_members(), 0)
        self.assertEqual(len(s.member_properties), 0)

        members = np.array([[0, 1, 2], [1, 2, 0]])
        s = Structure(nodes, members, label='label')
        self.assertEqual(s.label, 'label')
        self.assertEqual(s.nodes.dtype, np.float_)
        self.assertEqual(s.members.dtype, np.int64)
        np.testing.assert_array_equal(s.nodes, nodes)
        self.assertEqual(len(s.node_properties), 3)
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.node_tags, {})
        np.testing.assert_array_equal(s.members, members)
        np.testing.assert_array_equal(s.member_tags['bar'],
                                      np.arange(0, 3, dtype=np.int64))
        np.testing.assert_array_equal(s.member_tags['string'], np.zeros((0,), np.int64))
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.get_number_of_members(), 3)
        self.assertEqual(len(s.member_properties), 3)

        s = Structure(nodes, members, number_of_strings=1)
        self.assertEqual(s.label, None)
        self.assertEqual(s.nodes.dtype, np.float_)
        self.assertEqual(s.members.dtype, np.int64)
        np.testing.assert_array_equal(s.nodes, nodes)
        self.assertEqual(len(s.node_properties), 3)
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.node_tags, {})
        np.testing.assert_array_equal(s.members, members)
        np.testing.assert_array_equal(s.member_tags['bar'],
                                      np.arange(1, 3, dtype=np.int64))
        np.testing.assert_array_equal(s.member_tags['string'],
                                      np.arange(0, 1, dtype=np.int64))
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.get_number_of_members(), 3)
        self.assertEqual(len(s.member_properties), 3)

        s = Structure(nodes, members, number_of_strings=2)
        self.assertEqual(s.label, None)
        self.assertEqual(s.nodes.dtype, np.float_)
        self.assertEqual(s.members.dtype, np.int64)
        np.testing.assert_array_equal(s.nodes, nodes)
        self.assertEqual(len(s.node_properties), 3)
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.node_tags, {})
        np.testing.assert_array_equal(s.members, members)
        np.testing.assert_array_equal(s.member_tags['bar'],
                                      np.arange(2, 3, dtype=np.int64))
        np.testing.assert_array_equal(s.member_tags['string'],
                                      np.arange(0, 2, dtype=np.int64))
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.get_number_of_members(), 3)
        self.assertEqual(len(s.member_properties), 3)

        member_defaults = Structure.member_defaults.copy()
        Structure.member_defaults['vertical'] = {'linewidth': 1001, 'volume': 2}
        s = Structure(nodes, members,
                      member_tags={
                          'bar': np.arange(1, 3, dtype=np.int64),
                          'string': np.arange(0, 1, dtype=np.int64),
                          'vertical': np.arange(2, 3, dtype=np.int64)
                      },
                      node_tags={
                          'bottom': np.array([0, 1], dtype=np.int64),
                          'top': np.array([2], dtype=np.int64)
                      })
        self.assertEqual(s.label, None)
        self.assertEqual(s.nodes.dtype, np.float_)
        self.assertEqual(s.members.dtype, np.int64)
        np.testing.assert_array_equal(s.nodes, nodes)
        self.assertEqual(len(s.node_properties), 3)
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(set(s.node_tags.keys()), {'bottom', 'top'})
        np.testing.assert_array_equal(s.node_tags['bottom'],
                                      np.array([0, 1], dtype=np.int64))
        np.testing.assert_array_equal(s.node_tags['top'], np.array([2], dtype=np.int64))
        np.testing.assert_array_equal(s.members, members)
        np.testing.assert_array_equal(s.member_tags['bar'],
                                      np.arange(1, 3, dtype=np.int64))
        np.testing.assert_array_equal(s.member_tags['string'],
                                      np.arange(0, 1, dtype=np.int64))
        np.testing.assert_array_equal(s.member_tags['vertical'],
                                      np.arange(2, 3, dtype=np.int64))
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.get_number_of_members(), 3)
        self.assertEqual(s.get_member_tags(0), ['string'])
        self.assertEqual(s.get_member_tags(1), ['bar'])
        self.assertEqual(s.get_member_tags(2), ['bar', 'vertical'])
        self.assertTrue(s.has_member_tag(2, 'bar'))
        self.assertTrue(s.has_member_tag(2, 'vertical'))
        self.assertFalse(s.has_member_tag(1, 'vertical'))
        np.testing.assert_array_equal(s.get_members_by_tag('bar'), [1, 2])
        np.testing.assert_array_equal(s.get_members_by_tag('bar', 'vertical'), [2])
        self.assertEqual(len(s.member_properties), 3)
        self.assertEqual(s.member_properties.loc[1, 'linewidth'], 2)
        self.assertEqual(s.member_properties.loc[2, 'linewidth'], 1001)
        self.assertEqual(s.get_member_properties([1, 2],
                                                 'volume', 'mass').to_dict('index'),
                         {1: {'volume': 0., 'mass': 1.}, 2: {'volume': 2., 'mass': 1.}})
        self.assertEqual(s.get_member_properties(2, 'volume', 'mass').to_dict(),
                         {'volume': 2., 'mass': 1.})
        Structure.member_defaults = member_defaults

    def test_get_set_member_properties(self):

        nodes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        members = np.array([[0, 1, 2], [1, 2, 0]])
        s = Structure(nodes, members,
                      member_tags={
                          'bar': np.arange(1, 3, dtype=np.int64),
                          'string': np.arange(0, 1, dtype=np.int64),
                          'vertical': np.arange(2, 3, dtype=np.int64)
                      },
                      node_tags={
                          'bottom': np.array([0, 1], dtype=np.int64),
                          'top': np.array([2], dtype=np.int64)
                      })

        properties = s.get_member_properties([1, 2], 'volume')
        self.assertIsInstance(properties, pd.Series)
        self.assertEqual(len(properties), 2)
        np.testing.assert_array_equal(properties.values, [0, 0])

        properties = s.get_member_properties(slice(None), 'volume')
        self.assertIsInstance(properties, pd.Series)
        self.assertEqual(len(properties), 3)

        properties = s.get_member_properties([1, 2], 'facecolor')
        self.assertIsInstance(properties, pd.Series)
        self.assertEqual(len(properties), 2)

        properties = s.get_member_properties(slice(None), 'facecolor')
        self.assertIsInstance(properties, pd.Series)
        self.assertEqual(len(properties), 3)

        properties = s.get_member_properties([1, 2], 'volume', 'facecolor')
        self.assertIsInstance(properties, pd.DataFrame)
        self.assertEqual(len(properties), 2)

        properties = s.get_member_properties(slice(None), 'volume', 'facecolor')
        self.assertIsInstance(properties, pd.DataFrame)
        self.assertEqual(len(properties), 3)

        s.set_member_properties(2, 'volume', 3)
        np.testing.assert_array_equal(s.member_properties['volume'].values, [0, 0, 3])

        s.set_member_properties([1, 2], 'volume', [1, 2], scalar=False)
        np.testing.assert_array_equal(s.member_properties['volume'].values, [0, 1, 2])

        s.set_member_properties([1, 2], 'volume', 3)
        np.testing.assert_array_equal(s.member_properties['volume'].values, [0, 3, 3])

        s.set_member_properties([1, 2], 'volume', 1)
        np.testing.assert_array_equal(s.member_properties['volume'].values, [0, 1, 1])

        s.set_member_properties([1, 2], 'volume', 2, 'mass', 4)
        np.testing.assert_array_equal(s.member_properties['volume'].values, [0, 2, 2])
        np.testing.assert_array_equal(s.member_properties['mass'].values, [1, 4, 4])

        s.set_member_properties(2, 'facecolor', utils.Colors.BROWN.value)
        self.assertEqual(s.member_properties['facecolor'].tolist(),
                         [utils.Colors.ORANGE.value,
                          utils.Colors.BLUE.value, utils.Colors.BROWN.value])

        s.set_member_properties([1, 2], 'facecolor', utils.Colors.GREEN.value)
        self.assertEqual(s.member_properties['facecolor'].tolist(),
                         [utils.Colors.ORANGE.value,
                          utils.Colors.GREEN.value, utils.Colors.GREEN.value])

        s.set_member_properties([1, 2], 'facecolor',
                                [utils.Colors.BLUE.value, utils.Colors.ORANGE.value],
                                scalar=False)
        self.assertEqual(s.member_properties['facecolor'].tolist(),
                         [utils.Colors.ORANGE.value,
                          utils.Colors.BLUE.value, utils.Colors.ORANGE.value])

    def test_add_members_and_add_nodes(self):

        nodes1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        members1 = np.array([[0, 1, 2], [1, 2, 0]])
        s = Structure(nodes1, members1,
                      member_tags={
                          'bar': np.arange(1, 3, dtype=np.int64),
                          'string': np.arange(0, 1, dtype=np.int64),
                          'vertical': np.arange(2, 3, dtype=np.int64)
                      })
        members2 = s.get_number_of_nodes() + np.array([[0], [1]])
        nodes2 = np.array([[1, 0], [1, 1], [0, 1]])
        s.add_nodes(nodes2, node_tags={
            'bottom': np.array([0], dtype=np.int64),
            'top': np.array([1], dtype=np.int64)
        })
        s.add_members(members2, number_of_strings=1)

        self.assertEqual(s.get_number_of_nodes(), 5)
        self.assertEqual(len(s.node_properties), 5)
        self.assertEqual(set(s.node_tags.keys()), {'bottom', 'top'})
        np.testing.assert_array_equal(s.node_tags['bottom'],
                                      np.array([3], dtype=np.int64))
        np.testing.assert_array_equal(s.node_tags['top'],
                                      np.array([4], dtype=np.int64))
        self.assertEqual(s.get_number_of_members(), 4)
        self.assertEqual(set(s.member_tags.keys()), {'bar', 'string', 'vertical'})
        np.testing.assert_array_equal(s.member_tags['bar'], [1, 2])
        np.testing.assert_array_equal(s.member_tags['string'], [0, 3])
        np.testing.assert_array_equal(s.member_tags['vertical'], [2])
        np.testing.assert_array_equal(s.members, [[0, 1, 2, 3], [1, 2, 0, 4]])
        np.testing.assert_array_equal(s.nodes,
                                      [[1, 0, 0, 1, 0],
                                       [0, 1, 0, 1, 1],
                                       [0, 0, 1, 0, 1]])
        np.testing.assert_array_equal(s.member_properties.index, np.arange(4))

    def test_copy(self):

        nodes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        members = np.array([[0, 1, 2], [1, 2, 0]])
        member_defaults = Structure.member_defaults.copy()
        Structure.member_defaults['vertical'] = {'linewidth': 1001, 'volume': 2}
        s = Structure(nodes, members,
                      member_tags={
                          'bar': np.arange(1, 3, dtype=np.int64),
                          'string': np.arange(0, 1, dtype=np.int64),
                          'vertical': np.arange(2, 3, dtype=np.int64)
                      },
                      node_tags={
                          'bottom': np.array([0, 1], dtype=np.int64),
                          'top': np.array([2], dtype=np.int64)
                      }, label='label')

        self.assertEqual(s.label, 'label')
        self.assertEqual(s.nodes.dtype, np.float_)
        self.assertEqual(s.members.dtype, np.int64)
        np.testing.assert_array_equal(s.nodes, nodes)
        self.assertEqual(len(s.node_properties), 3)
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(set(s.node_tags.keys()), {'bottom', 'top'})
        np.testing.assert_array_equal(s.node_tags['bottom'],
                                      np.array([0, 1], dtype=np.int64))
        np.testing.assert_array_equal(s.node_tags['top'],
                                      np.array([2], dtype=np.int64))
        np.testing.assert_array_equal(s.members, members)
        np.testing.assert_array_equal(s.member_tags['bar'],
                                      np.arange(1, 3, dtype=np.int64))
        np.testing.assert_array_equal(s.member_tags['string'],
                                      np.arange(0, 1, dtype=np.int64))
        np.testing.assert_array_equal(s.member_tags['vertical'],
                                      np.arange(2, 3, dtype=np.int64))
        self.assertEqual(s.get_number_of_nodes(), 3)
        self.assertEqual(s.get_number_of_members(), 3)
        self.assertEqual(s.get_member_tags(0), ['string'])
        self.assertEqual(s.get_member_tags(1), ['bar'])
        self.assertEqual(s.get_member_tags(2), ['bar', 'vertical'])
        self.assertTrue(s.has_member_tag(2, 'bar'))
        self.assertTrue(s.has_member_tag(2, 'vertical'))
        self.assertFalse(s.has_member_tag(1, 'vertical'))
        np.testing.assert_array_equal(s.get_members_by_tag('bar'), [1, 2])
        np.testing.assert_array_equal(s.get_members_by_tag('bar', 'vertical'), [2])
        self.assertEqual(len(s.member_properties), 3)
        self.assertEqual(s.member_properties.loc[1, 'linewidth'], 2)
        self.assertEqual(s.member_properties.loc[2, 'linewidth'], 1001)
        self.assertEqual(s.get_member_properties([1, 2],
                                                 'volume', 'mass').to_dict('index'),
                         {1: {'volume': 0., 'mass': 1.}, 2: {'volume': 2., 'mass': 1.}})
        self.assertEqual(s.get_member_properties(2, 'volume', 'mass').to_dict(),
                         {'volume': 2., 'mass': 1.})

        copy = s.copy()

        self.assertEqual(copy.label, 'label')
        self.assertEqual(copy.nodes.dtype, np.float_)
        self.assertEqual(copy.members.dtype, np.int64)
        np.testing.assert_array_equal(copy.nodes, nodes)
        self.assertEqual(len(copy.node_properties), 3)
        self.assertEqual(copy.get_number_of_nodes(), 3)
        self.assertEqual(set(copy.node_tags.keys()), {'bottom', 'top'})
        np.testing.assert_array_equal(copy.node_tags['bottom'],
                                      np.array([0, 1], dtype=np.int64))
        np.testing.assert_array_equal(copy.node_tags['top'],
                                      np.array([2], dtype=np.int64))
        np.testing.assert_array_equal(copy.members, members)
        np.testing.assert_array_equal(copy.member_tags['bar'],
                                      np.arange(1, 3, dtype=np.int64))
        np.testing.assert_array_equal(copy.member_tags['string'],
                                      np.arange(0, 1, dtype=np.int64))
        np.testing.assert_array_equal(copy.member_tags['vertical'],
                                      np.arange(2, 3, dtype=np.int64))
        self.assertEqual(copy.get_number_of_nodes(), 3)
        self.assertEqual(copy.get_number_of_members(), 3)
        self.assertEqual(copy.get_member_tags(0), ['string'])
        self.assertEqual(copy.get_member_tags(1), ['bar'])
        self.assertEqual(copy.get_member_tags(2), ['bar', 'vertical'])
        self.assertTrue(copy.has_member_tag(2, 'bar'))
        self.assertTrue(copy.has_member_tag(2, 'vertical'))
        self.assertFalse(copy.has_member_tag(1, 'vertical'))
        np.testing.assert_array_equal(copy.get_members_by_tag('bar'), [1, 2])
        np.testing.assert_array_equal(copy.get_members_by_tag('bar', 'vertical'), [2])
        self.assertEqual(len(copy.member_properties), 3)
        self.assertEqual(copy.member_properties.loc[1, 'linewidth'], 2)
        self.assertEqual(copy.member_properties.loc[2, 'linewidth'], 1001)
        self.assertEqual(copy.get_member_properties([1, 2],
                                                    'volume', 'mass').to_dict('index'),
                         {1: {'volume': 0., 'mass': 1.}, 2: {'volume': 2., 'mass': 1.}})
        self.assertEqual(copy.get_member_properties(2, 'volume', 'mass').to_dict(),
                         {'volume': 2., 'mass': 1.})

        Structure.member_defaults = member_defaults

    def test_merge(self):

        nodes1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        members1 = np.array([[0, 1, 2], [1, 2, 0]])
        s1 = Structure(nodes1, members1,
                       member_tags={
                           'bar': np.arange(1, 3, dtype=np.int64),
                           'string': np.arange(0, 1, dtype=np.int64),
                           'vertical': np.arange(2, 3, dtype=np.int64)
                       },
                       node_tags={
                           'bottom': np.array([0, 1], dtype=np.int64),
                           'top': np.array([2], dtype=np.int64)
                       })

        nodes2 = np.array([[1, 0], [1, 1], [0, 1]])
        members2 = np.array([[0], [1]])
        s2 = Structure(nodes2, members2, number_of_strings=1)

        s1.merge(s2)

        self.assertEqual(s1.get_number_of_nodes(), 5)
        self.assertEqual(len(s1.node_properties), 5)
        self.assertEqual(set(s1.node_tags.keys()), {'bottom', 'top'})
        np.testing.assert_array_equal(s1.node_tags['bottom'],
                                      np.array([0, 1], dtype=np.int64))
        np.testing.assert_array_equal(s1.node_tags['top'],
                                      np.array([2], dtype=np.int64))
        self.assertEqual(s1.get_number_of_members(), 4)
        self.assertEqual(set(s1.member_tags.keys()), {'bar', 'string', 'vertical'})
        np.testing.assert_array_equal(s1.member_tags['bar'], [1, 2])
        np.testing.assert_array_equal(s1.member_tags['string'], [0, 3])
        np.testing.assert_array_equal(s1.member_tags['vertical'], [2])
        np.testing.assert_array_equal(s1.members, [[0, 1, 2, 3], [1, 2, 0, 4]])
        np.testing.assert_array_equal(s1.nodes,
                                      [[1, 0, 0, 1, 0],
                                       [0, 1, 0, 1, 1],
                                       [0, 0, 1, 0, 1]])
        np.testing.assert_array_equal(s1.member_properties.index, np.arange(4))

        # nodes1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # members1 = np.array([[0, 1, 2], [1, 2, 0]])
        s1 = Structure(nodes1, members1,
                       member_tags={
                           'bar': np.arange(1, 3, dtype=np.int64),
                           'string': np.arange(0, 1, dtype=np.int64),
                           'vertical': np.arange(2, 3, dtype=np.int64)
                       },
                       node_tags={
                           'bottom': np.array([0, 1], dtype=np.int64),
                           'top': np.array([2], dtype=np.int64)
                       })

        # nodes2 = np.array([[1, 0], [1, 1], [0, 1]])
        # members2 = np.array([[0], [1]])
        s2 = Structure(nodes2, members2, number_of_strings=1)

        s2.merge(s1)
        self.assertEqual(s2.get_number_of_nodes(), 5)
        self.assertEqual(len(s2.node_properties), 5)
        self.assertEqual(set(s2.node_tags.keys()), {'bottom', 'top'})
        np.testing.assert_array_equal(s2.node_tags['bottom'],
                                      np.array([2, 3], dtype=np.int64))
        np.testing.assert_array_equal(s2.node_tags['top'],
                                      np.array([4], dtype=np.int64))
        self.assertEqual(s2.get_number_of_members(), 4)
        self.assertEqual(set(s2.member_tags.keys()), {'bar', 'string', 'vertical'})
        np.testing.assert_array_equal(s2.member_tags['bar'], [2, 3])
        np.testing.assert_array_equal(s2.member_tags['string'], [0, 1])
        np.testing.assert_array_equal(s2.member_tags['vertical'], [3])
        np.testing.assert_array_equal(s2.members, [[0, 2, 3, 4], [1, 3, 4, 2]])
        np.testing.assert_array_equal(s2.nodes,
                                      [[1, 0, 1, 0, 0],
                                       [1, 1, 0, 1, 0],
                                       [0, 1, 0, 0, 1]])
        np.testing.assert_array_equal(s2.member_properties.index, np.arange(4))

        s1 = Structure(nodes1, members1,
                       member_tags={
                           'bar': np.arange(1, 3, dtype=np.int64),
                           'string': np.arange(0, 1, dtype=np.int64),
                           'vertical': np.arange(2, 3, dtype=np.int64)
                       },
                       node_tags={
                           'bottom': np.array([0, 1], dtype=np.int64),
                           'top': np.array([2], dtype=np.int64)
                       })

        # nodes2 = np.array([[1, 0], [1, 1], [0, 1]])
        # members2 = np.array([[0], [1]])
        s2 = Structure(nodes2, members2, number_of_strings=1)

        s = structure.merge(s1, s2)

        self.assertFalse(s is s1)
        self.assertFalse(s is s2)
        self.assertEqual(s.get_number_of_nodes(), 5)
        self.assertEqual(len(s.node_properties), 5)
        self.assertEqual(set(s.node_tags.keys()), {'bottom', 'top'})
        np.testing.assert_array_equal(s.node_tags['bottom'],
                                      np.array([0, 1], dtype=np.int64))
        np.testing.assert_array_equal(s.node_tags['top'],
                                      np.array([2], dtype=np.int64))
        self.assertEqual(s.get_number_of_members(), 4)
        self.assertEqual(set(s.member_tags.keys()), {'bar', 'string', 'vertical'})
        np.testing.assert_array_equal(s.member_tags['bar'], [1, 2])
        np.testing.assert_array_equal(s.member_tags['string'], [0, 3])
        np.testing.assert_array_equal(s.member_tags['vertical'], [2])
        np.testing.assert_array_equal(s.members, [[0, 1, 2, 3], [1, 2, 0, 4]])
        np.testing.assert_array_equal(s.nodes,
                                      [[1, 0, 0, 1, 0],
                                       [0, 1, 0, 1, 1],
                                       [0, 0, 1, 0, 1]])
        np.testing.assert_array_equal(s.member_properties.index, np.arange(4))

    def test_length_com_cog_translate(self):

        nodes1 = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 2, 1]])
        members1 = np.array([[0, 1, 2, 3], [1, 2, 0, 0]])
        s = Structure(nodes1, members1, number_of_strings=2)

        # test get member length
        np.testing.assert_array_equal(s.get_member_length(),
                                      [np.linalg.norm(nodes1[:, 1]-nodes1[:, 0]),
                                       np.linalg.norm(nodes1[:, 2]-nodes1[:, 1]),
                                       np.linalg.norm(nodes1[:, 0]-nodes1[:, 2]),
                                       np.linalg.norm(nodes1[:, 3]-nodes1[:, 0])])

        # test center of mass
        np.testing.assert_array_equal(s.get_center_of_mass(),
                                      (nodes1[:, 1] + nodes1[:, 0] +
                                       nodes1[:, 2] + nodes1[:, 1] +
                                       nodes1[:, 0] + nodes1[:, 2] +
                                       nodes1[:, 3] + nodes1[:, 0])/8)

        # test center of gravity
        np.testing.assert_array_equal(s.get_centroid(),
                                      (nodes1[:, 0] + nodes1[:, 1] +
                                       nodes1[:, 2] + nodes1[:, 3])/4)

    def test_rotate(self):

        nodes1 = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 2, 1]])
        members1 = np.array([[0, 1, 2, 3], [1, 2, 0, 0]])
        s = Structure(nodes1, members1, number_of_strings=2)

        v = np.array([1, 1, 1])
        s.rotate(v)
        R = scipy.spatial.transform.Rotation.from_rotvec(v)
        np.testing.assert_array_equal(s.nodes, R.apply(nodes1.transpose()).transpose())

    def test_reflect(self):

        nodes1 = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 2, 1]])
        members1 = np.array([[0, 1, 2, 3], [1, 2, 0, 0]])
        s = Structure(nodes1, members1, number_of_strings=2)

        v = np.array([1, 1, 1])
        s.reflect(v)
        H = np.eye(3) - 2 * np.outer(v, v)/(np.linalg.norm(v)**2)
        np.testing.assert_array_equal(s.nodes, H @ nodes1)

        s.set_nodes(nodes1)
        p = np.array([1, -1, 2])
        s.reflect(v, p)
        p = p.reshape((3, 1))
        np.testing.assert_array_equal(s.nodes, (H @ (nodes1 - p)) + p)

    def test_remove_nodes(self):

        nodes1 = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 2, 1]])
        nodes1_tags = {'tags': np.array([0, 2, 3], dtype=np.int64)}
        members1 = np.array([[0, 1, 2], [1, 2, 0]])
        s = Structure(nodes1, members1, number_of_strings=2, node_tags=nodes1_tags)

        np.testing.assert_array_equal(s.get_unused_nodes(), [3])
        self.assertTrue(s.has_unused_nodes())

        s.remove_nodes([3])
        np.testing.assert_array_equal(s.nodes, nodes1[:, :3])
        self.assertEqual(len(s.node_properties), 3)
        np.testing.assert_array_equal(s.node_properties.index, np.arange(3))
        np.testing.assert_array_equal(s.members, members1)
        np.testing.assert_array_equal(s.node_tags['tags'], [0, 2])
        self.assertFalse(s.has_unused_nodes())

        members2 = np.array([[0, 1, 3], [1, 3, 0]])
        s = Structure(nodes1, members2, number_of_strings=2, node_tags=nodes1_tags)

        np.testing.assert_array_equal(s.get_unused_nodes(), [2])
        self.assertTrue(s.has_unused_nodes())

        s.remove_nodes([2])
        np.testing.assert_array_equal(s.nodes, nodes1[:, [0, 1, 3]])
        self.assertEqual(len(s.node_properties), 3)
        np.testing.assert_array_equal(s.node_properties.index, np.arange(3))
        np.testing.assert_array_equal(s.node_tags['tags'], [0, 2])
        np.testing.assert_array_equal(s.members, members1)

        members3 = np.array([[0, 3], [3, 0]])
        s = Structure(nodes1, members3, number_of_strings=2, node_tags=nodes1_tags)

        np.testing.assert_array_equal(s.get_unused_nodes(), [1, 2])
        self.assertTrue(s.has_unused_nodes())

        s.remove_nodes([1, 2])
        np.testing.assert_array_equal(s.nodes, nodes1[:, [0, 3]])
        np.testing.assert_array_equal(s.members, np.array([[0, 1], [1, 0]]))
        np.testing.assert_array_equal(s.node_tags['tags'], [0, 1])
        self.assertFalse(s.has_unused_nodes())

        members4 = np.array([[1, 3], [3, 1]])
        s = Structure(nodes1, members4, number_of_strings=2, node_tags=nodes1_tags)

        np.testing.assert_array_equal(s.get_unused_nodes(), [0, 2])
        self.assertTrue(s.has_unused_nodes())

        s.remove_nodes([0, 2])
        np.testing.assert_array_equal(s.nodes, nodes1[:, [1, 3]])
        self.assertEqual(len(s.node_properties), 2)
        np.testing.assert_array_equal(s.node_properties.index, np.arange(2))
        np.testing.assert_array_equal(s.members, np.array([[0, 1], [1, 0]]))
        np.testing.assert_array_equal(s.node_tags['tags'], [1])
        self.assertFalse(s.has_unused_nodes())

        # has used nodes
        s = Structure(nodes1, members1, number_of_strings=2, node_tags=nodes1_tags)

        np.testing.assert_array_equal(s.get_unused_nodes(), [3])
        self.assertTrue(s.has_unused_nodes())

        with self.assertWarns(Warning):
            s.remove_nodes([2, 3])

        np.testing.assert_array_equal(s.nodes, nodes1[:, :3])
        self.assertEqual(len(s.node_properties), 3)
        np.testing.assert_array_equal(s.node_properties.index, np.arange(3))
        np.testing.assert_array_equal(s.members, members1)
        np.testing.assert_array_equal(s.node_tags['tags'], [0, 2])
        self.assertFalse(s.has_unused_nodes())

        s = Structure(nodes1, members1, number_of_strings=2, node_tags=nodes1_tags)

        np.testing.assert_array_equal(s.get_unused_nodes(), [3])
        self.assertTrue(s.has_unused_nodes())

        with self.assertWarns(Warning):
            s.remove_nodes([1, 2])

        np.testing.assert_array_equal(s.nodes, nodes1)
        self.assertEqual(len(s.node_properties), 4)
        np.testing.assert_array_equal(s.node_properties.index, np.arange(4))
        np.testing.assert_array_equal(s.members, members1)
        np.testing.assert_array_equal(s.node_tags['tags'], [0, 2, 3])
        self.assertTrue(s.has_unused_nodes())

        # None as parameter

        members3 = np.array([[0, 3], [3, 0]])
        s = Structure(nodes1, members3, number_of_strings=2, node_tags=nodes1_tags)

        np.testing.assert_array_equal(s.get_unused_nodes(), [1, 2])
        self.assertTrue(s.has_unused_nodes())

        s.remove_nodes()
        np.testing.assert_array_equal(s.nodes, nodes1[:, [0, 3]])
        self.assertEqual(len(s.node_properties), 2)
        np.testing.assert_array_equal(s.node_properties.index, np.arange(2))
        np.testing.assert_array_equal(s.members, np.array([[0, 1], [1, 0]]))
        np.testing.assert_array_equal(s.node_tags['tags'], [0, 1])
        self.assertFalse(s.has_unused_nodes())

    def test_remove_members(self):

        nodes = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 2, 1]])
        nodes_tags = {'tags': np.array([0, 2, 3], dtype=np.int64)}
        members = np.array([[0, 1, 2], [1, 2, 0]])
        members_tag = {'mtag': np.array([0, 2], dtype=np.int64)}

        s = Structure(nodes, members, number_of_strings=2,
                      node_tags=nodes_tags, member_tags=members_tag)

        s.remove_members([2])
        np.testing.assert_array_equal(s.nodes, nodes)
        self.assertEqual(len(s.node_properties), 4)
        np.testing.assert_array_equal(s.members, members[:, :2])
        np.testing.assert_array_equal(s.member_tags['mtag'], [0])

        s = Structure(nodes, members, number_of_strings=2,
                      node_tags=nodes_tags, member_tags=members_tag)

        s.remove_members([0, 1])
        np.testing.assert_array_equal(s.nodes, nodes)
        self.assertEqual(len(s.node_properties), 4)
        np.testing.assert_array_equal(s.members, members[:, [2]])
        np.testing.assert_array_equal(s.member_tags['mtag'], [0])

        s = Structure(nodes, members, number_of_strings=2,
                      node_tags=nodes_tags, member_tags=members_tag)

        s.remove_members([1, 2])
        np.testing.assert_array_equal(s.nodes, nodes)
        self.assertEqual(len(s.node_properties), 4)
        np.testing.assert_array_equal(s.members, members[:, [0]])
        np.testing.assert_array_equal(s.member_tags['mtag'], [0])

    def test_close_node(self):

        nodes1 = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 2, 1]])
        members1 = np.array([[0, 1, 2], [1, 2, 0]])
        s = Structure(nodes1, members1, number_of_strings=2)
        merge, merge_map = s.get_close_nodes()
        self.assertFalse(merge)

        s.merge_close_nodes()
        self.assertEqual(s.get_number_of_nodes(), 4)
        np.testing.assert_array_equal(s.nodes, nodes1)
        np.testing.assert_array_equal(s.members, members1)

        nodes1 = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 2, 0]])
        members1 = np.array([[0, 1, 2, 3], [1, 2, 0, 1]])
        s = Structure(nodes1, members1, number_of_strings=2)
        merge, merge_map = s.get_close_nodes()
        self.assertEqual(merge, {3})
        np.testing.assert_array_equal(merge_map, [0, 1, 2, 0])

        s.merge_close_nodes()
        self.assertEqual(s.get_number_of_nodes(), 3)
        np.testing.assert_array_equal(s.nodes, nodes1[:, :3])
        np.testing.assert_array_equal(s.members, np.array([[0, 1, 2, 0], [1, 2, 0, 1]]))

        nodes1 = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 2]])
        members1 = np.array([[0, 1, 2, 3], [1, 2, 0, 1]])
        s = Structure(nodes1, members1, number_of_strings=2)
        merge, merge_map = s.get_close_nodes()
        self.assertEqual(merge, {2})
        np.testing.assert_array_equal(merge_map, [0, 1, 0, 3])

        s.merge_close_nodes()
        self.assertEqual(s.get_number_of_nodes(), 3)
        np.testing.assert_array_equal(s.nodes, nodes1[:, [0, 1, 3]])
        np.testing.assert_array_equal(s.members, np.array([[0, 1, 0, 2], [1, 0, 0, 1]]))

        radius = 0.1
        nodes1 = np.array([[1, 0, 0, 1, 1],
                           [0, 1, 0, 0, 0],
                           [0, 0, 2, .9*radius, 1.8*radius]])
        members1 = np.array([[0, 1, 2, 3, 2], [1, 2, 0, 1, 4]])
        s = Structure(nodes1, members1, number_of_strings=2)
        merge, merge_map = s.get_close_nodes(radius)
        self.assertEqual(merge, {3, 4})
        np.testing.assert_array_equal(merge_map, [0, 1, 2, 0, 0])

        s.merge_close_nodes(radius)
        self.assertEqual(s.get_number_of_nodes(), 3)
        np.testing.assert_array_equal(s.nodes, nodes1[:, :3])
        np.testing.assert_array_equal(s.members,
                                      np.array([[0, 1, 2, 0, 2], [1, 2, 0, 1, 0]]))

    def test_equilibrium_update_member_properties(self):

        nodes = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        members = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]).transpose()
        s0 = Structure(nodes, members)

        center = s0.get_centroid()
        s0.add_nodes(center.reshape((3, 1)))

        s1 = s0.copy()

        members = np.array([[0, 4], [1, 4], [2, 4]]).transpose()
        s1.add_members(members, number_of_strings=3)

        # won't be in equilibrium with only three strings
        self.assertRaises(Exception, s1.equilibrium)

        s1 = s0.copy()

        members = np.array([[0, 4], [1, 4], [2, 4], [3, 4]]).transpose()
        s1.add_members(members, number_of_strings=4)

        # equilibrium
        s1.equilibrium()

        # set properties
        s1.update_member_properties()

        # forces
        force = np.hstack((np.kron(np.array([-np.sqrt(2), -1, np.sqrt(11)]),
                                   np.ones((3,))), [np.sqrt(3)]))
        np.testing.assert_allclose(s1.member_properties['force'], force)

        lambda_ = np.hstack((np.kron(np.array([-1, -1, 4]), np.ones((3,))), [4]))
        np.testing.assert_allclose(s1.member_properties['lambda_'], lambda_)

        mass = np.hstack((157./200. * np.ones((6,)), 157./200./4 * np.ones((4,))))
        np.testing.assert_allclose(
            s1.member_properties['mass'] / s1.get_member_length() / np.pi,
            mass)

        volume = np.hstack((np.ones((6,)), 1/4 * np.ones((4,))))
        np.testing.assert_allclose(
            10000 * s1.member_properties['volume'] / s1.get_member_length() / np.pi,
            volume)

        np.testing.assert_allclose(
            s1.member_properties['stiffness'],
            np.pi * s1.member_properties['modulus'] *
            (s1.member_properties['radius']**2 -
             s1.member_properties['inner_radius']**2)
            / s1.get_member_length())

        np.testing.assert_allclose(
            s1.member_properties['rest_length'],
            s1.get_member_length() * (1 - s1.member_properties['lambda_'].values
                                      / s1.member_properties['stiffness']))

    def test_node_tags(self):

        nodes = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 1], [0, 0, 2, 1, 3]])
        nodes_tags = {
            'nags0': np.array([0, 3], dtype=np.int64),
            'nags1': np.array([1, 2], dtype=np.int64)
        }
        members = np.array([[0, 1, 2, 3], [1, 2, 0, 4]])
        members_tag = {
            'mags0': np.array([0], dtype=np.int64),
            'mags1': np.array([1, 2], dtype=np.int64)
        }
        s = Structure(nodes, members,
                      number_of_strings=2,
                      node_tags=nodes_tags, member_tags=members_tag)

        s.add_node_tag('nags2', [0, 3])
        np.testing.assert_array_equal(s.node_tags['nags2'], [0, 3])

        s.add_node_tag('nags1', [3])
        np.testing.assert_array_equal(s.node_tags['nags1'], [1, 2, 3])

        s.remove_node_tag('nags1', [2])
        np.testing.assert_array_equal(s.node_tags['nags1'], [1, 3])

        s.delete_node_tag('nags1')
        self.assertFalse('nags1' in s.node_tags)

    def test_member_tags(self):

        nodes = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 1], [0, 0, 2, 1, 3]])
        nodes_tags = {
            'nags0': np.array([0, 3], dtype=np.int64),
            'nags1': np.array([1, 2], dtype=np.int64)
        }
        members = np.array([[0, 1, 2, 3], [1, 2, 0, 4]])
        members_tag = {
            'mags0': np.array([0], dtype=np.int64),
            'mags1': np.array([1, 2], dtype=np.int64)
        }
        s = Structure(nodes, members,
                      number_of_strings=2,
                      node_tags=nodes_tags, member_tags=members_tag)

        s.add_member_tag('mags2', [0, 3])
        np.testing.assert_array_equal(s.member_tags['mags2'], [0, 3])

        s.add_member_tag('mags1', [3])
        np.testing.assert_array_equal(s.member_tags['mags1'], [1, 2, 3])

        s.remove_member_tag('mags1', [2])
        np.testing.assert_array_equal(s.member_tags['mags1'], [1, 3])

        s.delete_member_tag('mags1')
        self.assertFalse('mags1' in s.member_tags)

    def test_node_constraint(self):

        nodes = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 1], [0, 0, 2, 1, 3]])
        nodes_tags = {
            'nags0': np.array([0, 3], dtype=np.int64),
            'nags1': np.array([1, 2], dtype=np.int64)
        }
        members = np.array([[0, 1, 2, 3], [1, 2, 0, 4]])
        members_tag = {
            'mags0': np.array([0], dtype=np.int64),
            'mags1': np.array([1, 2], dtype=np.int64)
        }
        s = Structure(nodes, members,
                      number_of_strings=2,
                      node_tags=nodes_tags, member_tags=members_tag)
        s.node_properties.loc[2, 'constraint'] = NodeConstraint()
        s.node_properties.loc[0, 'constraint'] = NodeConstraint(np.array(([[1, 1, 0]])))

        # print(s.node_properties.loc[2, 'constraint'])
        # print(s.node_properties.loc[2, 'constraint'].__class__)
        # print(s.node_properties.loc[0, 'constraint'])
        # print(s.node_properties.loc[0, 'constraint'].__class__)

        R, T = NodeConstraint.node_constraint(s.nodes, s.node_properties['constraint'])
        self.assertEqual(R.shape, (4, 15))
        self.assertEqual(T.shape, (15, 11))

    def test_dependent_nodes(self):

        nodes = np.array([[1, 0, 0, 0, 1, 1/2],
                          [0, 1, 0, 1, 1, 1/2],
                          [0, 0, 2, 1, 3, 0]])
        nodes_tags = {
            'nags0': np.array([0, 3], dtype=np.int64),
            'nags1': np.array([1, 2], dtype=np.int64)
        }
        members = np.array([[0, 1, 2, 3, 0, 1], [1, 2, 0, 4, 5, 5]])
        members_tag = {
            'mags0': np.array([0], dtype=np.int64),
            'mags1': np.array([1, 2, 5], dtype=np.int64)
        }
        s = Structure(nodes, members,
                      number_of_strings=2,
                      node_tags=nodes_tags, member_tags=members_tag)
        np.testing.assert_array_equal(s.get_members_per_node(), [3, 3, 2, 1, 1, 2])
        np.testing.assert_array_equal(s.get_colinear_nodes(), [5])
        dep_nodes, dep_members = s.get_colinear_nodes(return_members=True)
        np.testing.assert_array_equal(dep_nodes, [5])
        np.testing.assert_array_equal(dep_members, [[4, 5]])

        s.merge_colinear_nodes()
        self.assertEqual(s.get_number_of_nodes(), 5)
        self.assertEqual(s.get_number_of_members(), 5)
        np.testing.assert_array_equal(s.nodes, nodes[:, :-1])
        np.testing.assert_array_equal(s.members,
                                      np.array([[0, 1, 2, 3, 0], [1, 2, 0, 4, 1]]))
        np.testing.assert_array_equal(s.member_tags['bar'], [2, 3, 4])
        np.testing.assert_array_equal(s.member_tags['mags1'], [1, 2, 4])

    def test_slack_members(self):

        nodes = np.array([[1, 0, 0, 0, 1, 1/2],
                          [0, 1, 0, 1, 1, 1/2], [0, 0, 2, 1, 3, 0]])
        nodes_tags = {
            'nags0': np.array([0, 3], dtype=np.int64),
            'nags1': np.array([1, 2], dtype=np.int64)
        }
        members = np.array([[0, 1, 2, 3, 0, 1], [1, 2, 0, 4, 5, 5]])
        members_tag = {
            'mags0': np.array([0], dtype=np.int64),
            'mags1': np.array([1, 2, 5], dtype=np.int64)
        }
        s = Structure(nodes, members,
                      number_of_strings=2,
                      node_tags=nodes_tags, member_tags=members_tag)
        s.member_properties.loc[[0, 3, 5], 'lambda_'] = [-1, 2, -3]
        np.testing.assert_array_equal(s.get_slack_members(), [1, 2, 4])
        s.set_member_properties([2], 'lambda_', 1e-9)
        np.testing.assert_array_equal(s.get_slack_members(), [1, 2, 4])
        s.set_member_properties([0], 'lambda_', 1e-9)
        s.remove_members(s.get_slack_members())
        self.assertEqual(s.get_number_of_members(), 2)


if __name__ == '__main__':
    unittest.main()
