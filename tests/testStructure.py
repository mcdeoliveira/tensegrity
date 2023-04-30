import unittest

import numpy as np
import scipy

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
        np.testing.assert_array_equal(s.get_members_by_tags(['bar']), [1, 2])
        np.testing.assert_array_equal(s.get_members_by_tags(['bar', 'vertical']), [2])
        self.assertEqual(len(s.member_properties), 3)
        self.assertEqual(s.member_properties.loc[1, 'linewidth'], 2)
        self.assertEqual(s.member_properties.loc[2, 'linewidth'], 1001)
        self.assertEqual(s.get_member_properties([1, 2], ['volume', 'mass']).to_dict('index'),
                         {1: {'volume': 0., 'mass': 1.}, 2: {'volume': 2., 'mass': 1.}})
        self.assertEqual(s.get_member_properties(2, ['volume', 'mass']).to_dict(), {'volume': 2., 'mass': 1.})

    def testAddMembersAndAddNodes(self):

        nodes1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        members1 = np.array([[0, 1, 2], [1, 2, 0]])
        s1 = Structure(nodes1, members1,
                       member_tags={
                           'bar': np.arange(1, 3, dtype=np.uint64),
                           'string': np.arange(0, 1, dtype=np.uint64),
                           'vertical': np.arange(2, 3, dtype=np.uint64)
                       })

        members2 = s1.get_number_of_nodes() + np.array([[0], [1]])
        nodes2 = np.array([[1, 0], [1, 1], [0, 1]])
        s1.add_nodes(nodes2)
        s1.add_members(members2, number_of_strings=1)

        self.assertEqual(s1.get_number_of_nodes(), 5)
        self.assertEqual(s1.get_number_of_members(), 4)
        self.assertEqual(set(s1.member_tags.keys()), {'bar', 'string', 'vertical'})
        np.testing.assert_array_equal(s1.member_tags['bar'], [1, 2])
        np.testing.assert_array_equal(s1.member_tags['string'], [0, 3])
        np.testing.assert_array_equal(s1.member_tags['vertical'], [2])
        np.testing.assert_array_equal(s1.members, [[0, 1, 2, 3], [1, 2, 0, 4]])
        np.testing.assert_array_equal(s1.nodes, [[1, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 1]])
        np.testing.assert_array_equal(s1.member_properties.index, np.arange(4))

    def testMerge(self):

        nodes1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        members1 = np.array([[0, 1, 2], [1, 2, 0]])
        s1 = Structure(nodes1, members1,
                       member_tags={
                           'bar': np.arange(1, 3, dtype=np.uint64),
                           'string': np.arange(0, 1, dtype=np.uint64),
                           'vertical': np.arange(2, 3, dtype=np.uint64)
                       })

        nodes2 = np.array([[1, 0], [1, 1], [0, 1]])
        members2 = np.array([[0], [1]])
        s2 = Structure(nodes2, members2, number_of_strings=1)

        s1.merge(s2)

        self.assertEqual(s1.get_number_of_nodes(), 5)
        self.assertEqual(s1.get_number_of_members(), 4)
        self.assertEqual(set(s1.member_tags.keys()), {'bar', 'string', 'vertical'})
        np.testing.assert_array_equal(s1.member_tags['bar'], [1, 2])
        np.testing.assert_array_equal(s1.member_tags['string'], [0, 3])
        np.testing.assert_array_equal(s1.member_tags['vertical'], [2])
        np.testing.assert_array_equal(s1.members, [[0, 1, 2, 3], [1, 2, 0, 4]])
        np.testing.assert_array_equal(s1.nodes, [[1, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 1]])
        np.testing.assert_array_equal(s1.member_properties.index, np.arange(4))

        nodes1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        members1 = np.array([[0, 1, 2], [1, 2, 0]])
        s1 = Structure(nodes1, members1,
                       member_tags={
                           'bar': np.arange(1, 3, dtype=np.uint64),
                           'string': np.arange(0, 1, dtype=np.uint64),
                           'vertical': np.arange(2, 3, dtype=np.uint64)
                       })

        nodes2 = np.array([[1, 0], [1, 1], [0, 1]])
        members2 = np.array([[0], [1]])
        s2 = Structure(nodes2, members2, number_of_strings=1)

        s2.merge(s1)
        self.assertEqual(s2.get_number_of_nodes(), 5)
        self.assertEqual(s2.get_number_of_members(), 4)
        self.assertEqual(set(s2.member_tags.keys()), {'bar', 'string', 'vertical'})
        np.testing.assert_array_equal(s2.member_tags['bar'], [2, 3])
        np.testing.assert_array_equal(s2.member_tags['string'], [0, 1])
        np.testing.assert_array_equal(s2.member_tags['vertical'], [3])
        np.testing.assert_array_equal(s2.members, [[0, 2, 3, 4], [1, 3, 4, 2]])
        np.testing.assert_array_equal(s2.nodes, [[1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [0, 1, 0, 0, 1]])
        np.testing.assert_array_equal(s2.member_properties.index, np.arange(4))

    def test_length_and_com_translate(self):

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
                                      (nodes1[:, 1] + nodes1[:, 0] + nodes1[:, 2] + nodes1[:, 1] +
                                       nodes1[:, 0] + nodes1[:, 2] + nodes1[:, 3] + nodes1[:, 0])/8)

    def test_rotate(self):

        nodes1 = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 2, 1]])
        members1 = np.array([[0, 1, 2, 3], [1, 2, 0, 0]])
        s = Structure(nodes1, members1, number_of_strings=2)

        v = np.array([1, 1, 1])
        s.rotate(v)
        R = scipy.spatial.transform.Rotation.from_rotvec(v)
        np.testing.assert_array_equal(s.nodes, R.apply(nodes1.transpose()).transpose())

    def test_remove_nodes(self):

        nodes1 = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 2, 1]])
        members1 = np.array([[0, 1, 2], [1, 2, 0]])
        s = Structure(nodes1, members1, number_of_strings=2)

        np.testing.assert_array_equal(s.get_unused_nodes(), [3])
        self.assertTrue(s.has_unused_nodes())

        s.remove_nodes([3])
        np.testing.assert_array_equal(s.nodes, nodes1[:, :3])
        np.testing.assert_array_equal(s.members, members1)
        self.assertFalse(s.has_unused_nodes())

        members2 = np.array([[0, 1, 3], [1, 3, 0]])
        s = Structure(nodes1, members2, number_of_strings=2)

        np.testing.assert_array_equal(s.get_unused_nodes(), [2])
        self.assertTrue(s.has_unused_nodes())

        s.remove_nodes([2])
        np.testing.assert_array_equal(s.nodes, nodes1[:, [0, 1, 3]])
        np.testing.assert_array_equal(s.members, members1)

        members3 = np.array([[0, 3], [3, 0]])
        s = Structure(nodes1, members3, number_of_strings=2)

        np.testing.assert_array_equal(s.get_unused_nodes(), [1, 2])
        self.assertTrue(s.has_unused_nodes())

        s.remove_nodes([1, 2])
        np.testing.assert_array_equal(s.nodes, nodes1[:, [0, 3]])
        np.testing.assert_array_equal(s.members, np.array([[0, 1], [1, 0]]))
        self.assertFalse(s.has_unused_nodes())

        members4 = np.array([[1, 3], [3, 1]])
        s = Structure(nodes1, members4, number_of_strings=2)

        np.testing.assert_array_equal(s.get_unused_nodes(), [0, 2])
        self.assertTrue(s.has_unused_nodes())

        s.remove_nodes([0, 2])
        np.testing.assert_array_equal(s.nodes, nodes1[:, [1, 3]])
        np.testing.assert_array_equal(s.members, np.array([[0, 1], [1, 0]]))
        self.assertFalse(s.has_unused_nodes())

        # has used nodes
        s = Structure(nodes1, members1, number_of_strings=2)

        np.testing.assert_array_equal(s.get_unused_nodes(), [3])
        self.assertTrue(s.has_unused_nodes())

        with self.assertWarns(Warning):
            s.remove_nodes([2, 3])

        np.testing.assert_array_equal(s.nodes, nodes1[:, :3])
        np.testing.assert_array_equal(s.members, members1)
        self.assertFalse(s.has_unused_nodes())

        s = Structure(nodes1, members1, number_of_strings=2)

        np.testing.assert_array_equal(s.get_unused_nodes(), [3])
        self.assertTrue(s.has_unused_nodes())

        with self.assertWarns(Warning):
            s.remove_nodes([1, 2])

        np.testing.assert_array_equal(s.nodes, nodes1)
        np.testing.assert_array_equal(s.members, members1)
        self.assertTrue(s.has_unused_nodes())

        # Non as parameter

        members3 = np.array([[0, 3], [3, 0]])
        s = Structure(nodes1, members3, number_of_strings=2)

        np.testing.assert_array_equal(s.get_unused_nodes(), [1, 2])
        self.assertTrue(s.has_unused_nodes())

        s.remove_nodes()
        np.testing.assert_array_equal(s.nodes, nodes1[:, [0, 3]])
        np.testing.assert_array_equal(s.members, np.array([[0, 1], [1, 0]]))
        self.assertFalse(s.has_unused_nodes())


if __name__ == '__main__':
    unittest.main()
