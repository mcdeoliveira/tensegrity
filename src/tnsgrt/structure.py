import collections
import itertools
import more_itertools
import warnings
from dataclasses import dataclass
from functools import reduce
from typing import Optional, Dict, get_type_hints, Union, List, Sequence, \
    Type, Iterable, Tuple, Set, Any
from collections import ChainMap, defaultdict

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy

from . import utils
from . import optim
from .stiffness import Stiffness, NodeConstraint
from .utils import is_colinear


class Property:
    """
    Base class for storing properties

    Derived classes should implement ``dataclass`` from ``dataclasses``
    """

    @classmethod
    def to_dataframe(cls: Type['Property'],
                     data: Union[list, tuple] = tuple()) -> pd.DataFrame:
        # setup member property as pandas dataframe
        hints = get_type_hints(cls)
        return pd.DataFrame(data=data, columns=list(hints.keys())).astype(dtype=hints)


class Structure:
    """
    A ``Structure`` object

    :param nodes: a 3 x n array representing the ``Structure``'s nodes
    :param members: a 3 x m array representing the ``Structure``'s members
    :param number_of_strings: the number of strings in ``Structure``
    :param node_tags: a dictionary with the node tags
    :param member_tags: a dictionary with the member tags
    :param label: the ``Structure``'s label
    """

    @dataclass
    class NodeProperty(Property):
        """
        Subclass representing properties of nodes
        """
        radius: float = 0.002
        visible: bool = True
        constraint: object = None
        facecolor: object = utils.Colors.BLUE.value
        edgecolor: object = utils.Colors.BLUE.value

    node_defaults = {
    }

    @dataclass
    class MemberProperty(Property):
        """
        Subclass representing properties of members
        """
        lambda_: float = 0.
        force: float = 0.
        stiffness: float = 0.
        volume: float = 0.
        radius: float = 0.01
        inner_radius: float = 0
        mass: float = 1.
        rest_length: float = 0.
        # ASTM A36 steel
        yld: float = 250e6
        density: float = 7.85e3
        modulus: float = 200e9
        visible: bool = True
        facecolor: object = (1, 0, 0)
        edgecolor: object = (1, 0, 0)
        linewidth: int = 2
        linestyle: str = '-'

    member_defaults = {
        'bar': {
            'facecolor': utils.Colors.BLUE.value,
            'edgecolor': utils.Colors.BLUE.value
        },
        'string': {
            'facecolor': utils.Colors.ORANGE.value,
            'edgecolor': utils.Colors.ORANGE.value,
            'radius': 0.005
        }
    }

    def __init__(self,
                 nodes: npt.ArrayLike = np.zeros((3, 0), np.float_),
                 members: npt.ArrayLike = np.zeros((2, 0), np.int64),
                 number_of_strings: int = 0,
                 node_tags: Optional[Dict[str, npt.NDArray[np.int64]]] = None,
                 member_tags: Optional[Dict[str, npt.NDArray[np.int64]]] = None,
                 label: str = None):
        # label
        self.label: Optional[str] = label
        # nodes
        self.nodes: npt.NDArray[np.float_] = np.zeros((3, 0), np.float_)
        self.node_tags: Dict[str, npt.NDArray[np.int64]] = {}
        self.node_properties: pd.DataFrame = Structure.NodeProperty.to_dataframe()
        # members
        self.members: npt.NDArray[np.int64] = np.zeros((2, 0), np.int64)
        self.member_tags: Dict[str, npt.NDArray[np.int64]] = {
            'bar': np.zeros((0,), np.int64),
            'string': np.zeros((0,), np.int64)
        }
        self.member_properties: pd.DataFrame = Structure.MemberProperty.to_dataframe()

        # add nodes
        self.add_nodes(nodes, node_tags)

        # add members
        self.add_members(members, number_of_strings, member_tags)

    def __repr__(self) -> str:
        """
        :return: short description of structure as a string
        """
        return "Structure " + (f" labeled '{self.label}'" if self.label else '') + \
            f"with {self.get_number_of_nodes()} nodes, " \
            f"{self.get_number_of_members_by_tag('bar')} bars " \
            f"and {self.get_number_of_members_by_tag('string')} strings"

    def copy(self) -> 'Structure':
        """
        :return: copy of the Structure
        """

        # instantiate copy of the current structure
        copy = self.__class__()

        # copy basic structure
        copy.label = self.label
        copy.nodes = self.nodes.copy()
        copy.node_tags = self.node_tags.copy()
        copy.node_properties = self.node_properties.copy()
        copy.members = self.members.copy()
        copy.member_tags = self.member_tags.copy()
        copy.member_properties = self.member_properties.copy()

        return copy

    def set_nodes(self, nodes: npt.ArrayLike) -> None:
        """
        Set nodes of the ``Structure``

        :param nodes: the nodes
        """
        # convert to array
        nodes = np.array(nodes, np.float_)

        # test dimensions
        assert np.all(nodes.shape == self.nodes.shape), \
            'nodes shape must match current shape'

        # set nodes
        self.nodes: npt.NDArray[np.float_] = nodes

    def add_nodes(self, nodes: npt.ArrayLike,
                  node_tags: Optional[Dict[str, npt.NDArray[np.int64]]] = None) -> None:
        """
        Add nodes to the ``Structure``

        :param nodes: the nodes to add
        :param node_tags: the node tags to add
        """
        # add nodes and tags to current structure

        # convert to array
        nodes = np.array(nodes, np.float_)

        # test dimensions
        assert nodes.shape[0] == 3, 'nodes must be a 3 x n array'

        # node tags
        number_of_new_nodes = nodes.shape[1]
        if node_tags is None:
            node_tags = {}
        else:
            # make sure node tags are unique
            for k, v in node_tags.items():
                node_tags[k] = np.unique(v)
                assert np.amin(v) >= 0, \
                    'node tag index must be greater or equal than zero'
                assert np.amax(v) < number_of_new_nodes, \
                    'node tag index must be less than number of new nodes'

        # new node properties
        number_of_nodes = self.get_number_of_nodes()
        # determine tags that have defaults
        tags_with_defaults = \
            list(set(node_tags.keys()) & set(Structure.node_defaults.keys()))
        # apply defaults
        new_node_properties = \
            [Structure.NodeProperty(**ChainMap(*[Structure.node_defaults[tag]
                                                 for tag in tags_with_defaults
                                                 if i in node_tags[tag]]))
             for i in range(nodes.shape[1])]

        # add new nodes
        self.nodes: npt.NDArray[np.float_] = np.hstack((self.nodes, nodes))

        # add node tags
        for k, v in node_tags.items():
            self.node_tags[k] = np.hstack((self.node_tags[k], number_of_nodes + v)) \
                if k in self.node_tags else number_of_nodes + v

        # add default node properties
        self.node_properties = \
            pd.concat((self.node_properties,
                       Structure.NodeProperty.to_dataframe(new_node_properties)),
                      ignore_index=True)

    def get_number_of_nodes(self) -> int:
        """
        :return: the number of nodes in ``Structure``
        """
        return self.nodes.shape[1]

    def translate(self, v: npt.NDArray) -> 'Structure':
        """
        Translate all nodes of the ``Structure`` by the 3D vector ``normal``

        :param v: the 3D translation vector
        :return: self
        """
        assert v.shape == (3,), 'normal must be a three dimensional vector'
        self.nodes += v.reshape((3, 1))
        return self

    def rotate(self, v: npt.NDArray) -> 'Structure':
        """
        Rotate all nodes of the ``Structure`` by the 3D vector ``v``

        :param v: the 3D rotation vector
        :return: self

        **Notes:**

        1. See :meth:`scipy.spatial.transform.Rotation.from_rotvec` for details
        """
        assert v.shape == (3,), 'normal must be a three dimensional vector'
        rotation = scipy.spatial.transform.Rotation.from_rotvec(v)
        self.nodes = rotation.apply(self.nodes.transpose()).transpose()
        return self

    def reflect(self, v: npt.NDArray, p: Optional[npt.NDArray] = None) -> 'Structure':
        """
        Reflects the structure about a plane normal to the vector `v`, passing through
        the point `p`. If no point is given, it defaults to the origin.

        :param v: the 3D normal vector
        :param p: the 3D origin vector
        :return: self
        """
        assert v.shape == (3,), 'normal must be a three dimensional vector'

        if p is not None:
            assert p.shape == (3,), 'p must be a three dimensional vector'
            # translate by p
            self.nodes -= p.reshape((3, 1))

        # normalize normal
        length = np.linalg.norm(v)
        if length < 1e-6:
            warnings.warn('norm of vector normal is too small, '
                          'reflection not performed')
            return self

        # calculate reflection matrix
        reflection_matrix = np.eye(3) - (2 / length**2) * np.outer(v, v)

        # transform nodes
        self.nodes = reflection_matrix @ self.nodes

        if p is not None:
            # translate back to p
            self.nodes += p.reshape((3, 1))

        return self

    def get_unused_nodes(self) -> npt.NDArray[np.int64]:
        """
        :return: an array with the indices of the unused nodes
        """
        # calculate nodes that are in use
        used_nodes = np.unique(self.members)
        # return unused nodes
        return np.setdiff1d(np.arange(self.get_number_of_nodes()),
                            used_nodes, assume_unique=True)

    def has_unused_nodes(self) -> bool:
        """
        :return: ``True`` if there are no unused nodes
        """
        return len(self.get_unused_nodes()) > 0

    def get_nodes_by_tag(self, *tag: str) -> npt.NDArray[np.int64]:
        """
        Return a list of node indices that have given tags

        :param \\*tag: the tag
        :return: list of node indices
        """
        if len(tag) == 0:
            return np.zeros((0,))
        elif len(tag) == 1:
            return self.node_tags.get(tag[0], np.zeros((0,)))
        else:
            return reduce(lambda a1, a2: np.intersect1d(a1, a2, assume_unique=True),
                          [v for k, v in self.node_tags.items() if k in tag])

    def get_number_of_nodes_by_tag(self, tag: str) -> int:
        """
        Return the number of members with tag ``tag``

        :param tag: the tags
        :return: the number of members
        """
        return len(self.node_tags.get(tag, []))

    def get_members_per_node(self) -> npt.NDArray:
        """
        :return: number of members connected to each node
        """
        return np.bincount(self.members.ravel(), minlength=self.get_number_of_nodes())

    def get_colinear_nodes(self, epsilon: float = 1e-8,
                           return_members: bool = False) -> \
            Union[npt.NDArray[np.int64],
                  Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        """
        :params epsilon: accuracy of co-linearity test
        :params return_members: if ``True`` returns tuple of member indices as well
        :return: an array with the indices of the co-linear nodes or
                 tuple with an array with the indices of the co-linear nodes and an
                 array with the member indices

        **Notes:**

        1. Co-linear nodes are nodes that have only 2 co-linear members connected to
           them
        """
        # find nodes connected to only 2 members
        dependent_nodes = np.nonzero(self.get_members_per_node() == 2)[0]
        # loop through members with dependent nodes
        colinear_nodes = []
        colinear_members = []
        for node in dependent_nodes:
            # get members
            member_indices = np.any(self.members == node, axis=0)
            node_indices = self.members[:, member_indices]
            members = self.nodes[:, node_indices[1]] - self.nodes[:, node_indices[0]]
            # is colinear if projection is small
            if is_colinear(members[:, 0], members[:, 1], epsilon):
                colinear_nodes.append(node)
                if return_members:
                    colinear_members.append(np.nonzero(member_indices)[0])
        # return only co-linear nodes
        if return_members:
            return np.array(colinear_nodes, dtype=np.int64), \
                np.array(colinear_members, dtype=np.int64)
        else:
            return np.array(colinear_nodes, dtype=np.int64)

    def remove_nodes(self, nodes_to_be_removed: Optional[npt.ArrayLike] = None,
                     verify_if_unused: bool = True, verbose: bool = False) -> None:
        """
        Remove nodes from structure

        :param nodes_to_be_removed: the indices of the nodes to be deleted; if ``None``,
                                    delete all currently unused nodes
        :param verify_if_unused: if ``True`` verifies if the nodes to be deleted are
                                 not in use
        :param verbose: if ``True`` warns of the nodes to be deleted
        """
        if nodes_to_be_removed is None:
            # delete all unused nodes
            unused_nodes_to_be_deleted = self.get_unused_nodes()
        elif verify_if_unused:
            # sort nodes to be deleted
            nodes_to_be_removed = np.unique(nodes_to_be_removed)
            # calculate nodes that are in use
            used_nodes = np.unique(self.members)
            # find unused nodes
            unused_nodes_to_be_deleted = np.setdiff1d(nodes_to_be_removed, used_nodes,
                                                      assume_unique=True)
            # warn if different
            number_of_used_nodes = \
                len(nodes_to_be_removed) - len(unused_nodes_to_be_deleted)
            if number_of_used_nodes:
                warnings.warn(f'{number_of_used_nodes} nodes are still '
                              f'in use and were not deleted')
                if verbose:
                    wn = np.intersect1d(nodes_to_be_removed, used_nodes,
                                        assume_unique=True)
                    warnings.warn('The following nodes will not be removed: ' 
                                  f'{wn}')
        else:
            # go ahead without verifying if nodes are unused
            # WARNING: this may result in orphan members!
            unused_nodes_to_be_deleted = nodes_to_be_removed
        # delete if there are any unused nodes
        if len(unused_nodes_to_be_deleted):
            if verbose:
                warnings.warn('The following nodes will be removed: '
                              f'{unused_nodes_to_be_deleted}')
            # create new node map
            node_index = \
                np.delete(np.arange(self.get_number_of_nodes()),
                          unused_nodes_to_be_deleted)
            new_node_map = np.zeros((self.get_number_of_nodes(),), dtype=np.int_)
            new_node_map[node_index] = \
                np.arange(self.get_number_of_nodes() - len(unused_nodes_to_be_deleted))
            # remove nodes
            self.nodes = np.delete(self.nodes, unused_nodes_to_be_deleted, axis=1)
            # remove node properties
            self.node_properties.drop(unused_nodes_to_be_deleted, inplace=True)
            self.node_properties.reset_index(inplace=True)
            # remove nodes from tags
            self.node_tags = \
                {k: new_node_map[np.setdiff1d(v, unused_nodes_to_be_deleted)]
                 for k, v in self.node_tags.items()}
            # apply new node map to members
            self.members = new_node_map[self.members]

    def add_members(self, members: npt.ArrayLike,
                    number_of_strings: Optional[int] = None,
                    member_tags: Optional[Dict[str, npt.NDArray[np.int64]]] = None)\
            -> None:
        """
        Add members and tags to current structure

        :param members: the members to be added
        :param number_of_strings: the number of strings; if not ``None``,  then the
                                  first `number_of_strings` members are tagged as
                                  'strings' and the remaining members as 'bars'
        :param member_tags: the new members' tags
        """

        # convert to array
        members = np.array(members, np.int64)

        # test dimensions
        assert members.shape[0] == 2, 'members must be a 2 x m array'

        # member tags
        if number_of_strings is None and member_tags is None:
            raise Exception('Either type or number of strings must be provided')

        number_of_new_members = members.shape[1]
        new_member_tags = {}
        if number_of_strings is not None:
            # number of strings given
            assert number_of_strings <= number_of_new_members, \
                'number of added strings must be less than number of added members'
            number_of_new_bars = number_of_new_members - number_of_strings
            new_member_tags = {
                'string': np.arange(0, number_of_strings, dtype=np.int64),
                'bar': number_of_strings + np.arange(0, number_of_new_bars,
                                                     dtype=np.int64)
            }

        # if tags were given
        if member_tags is not None:
            # make sure member tags are unique
            for k, v in member_tags.items():
                member_tags[k] = np.unique(v)
                assert np.amin(v) >= 0, \
                    'member tag index must be greater or equal than zero'
                assert np.amax(v) < number_of_new_members, \
                    'member index must be less than number of members'
            # make sure bars and strings are mutually exclusive
            assert 'bar' not in member_tags or 'string' not in member_tags or \
                   np.intersect1d(member_tags['bar'], member_tags['string'],
                                  assume_unique=True).size == 0, \
                'bar and string tags must be mutually exclusive'
            # update member_tags
            new_member_tags.update(member_tags)

        # new member properties
        number_of_members = self.get_number_of_members()
        # determine tags that have defaults
        tags_with_defaults = \
            list(set(new_member_tags.keys()) & set(Structure.member_defaults.keys()))
        # apply defaults
        new_member_properties = \
            [Structure.MemberProperty(**ChainMap(*[Structure.member_defaults[tag]
                                                   for tag in tags_with_defaults
                                                   if i in new_member_tags[tag]]))
             for i in range(members.shape[1])]

        # make sure member index is valid
        number_of_nodes = self.get_number_of_nodes()
        assert number_of_new_members == 0 or np.amin(members) >= 0, \
            'member index must be greater or equal than zero'
        assert number_of_new_members == 0 or np.amax(members) < number_of_nodes, \
            'member index must be less than number of nodes'

        # add new members
        self.members = np.hstack((self.members, members))
        # add member tags
        for k, v in new_member_tags.items():
            self.member_tags[k] = \
                np.hstack((self.member_tags[k], number_of_members + v)) \
                if k in self.member_tags else number_of_members + v

        # add default member properties
        self.member_properties = \
            pd.concat((self.member_properties,
                       Structure.MemberProperty.to_dataframe(new_member_properties)),
                      ignore_index=True)

    def remove_members(self, members_to_be_deleted: Optional[npt.ArrayLike] = None,
                       verbose: bool = False):
        """
        Remove members from ``Structure``

        :param members_to_be_deleted: the indices of the members to be deleted
        :param verbose: if ``True`` warns of the members to be deleted
        :return:
        """
        if members_to_be_deleted is not None and len(members_to_be_deleted):
            if verbose:
                warnings.warn('The following members will be removed: '
                              f'{members_to_be_deleted}')
            # create new member map
            member_index = \
                np.delete(np.arange(self.get_number_of_members()),
                          members_to_be_deleted)
            new_members_map = np.zeros((self.get_number_of_members(),), dtype=np.int_)
            new_members_map[member_index] = \
                np.arange(self.get_number_of_members() - len(members_to_be_deleted))
            # remove members
            self.members = np.delete(self.members, members_to_be_deleted, axis=1)
            # remove member properties
            self.member_properties.drop(members_to_be_deleted, inplace=True)
            self.member_properties.reset_index(inplace=True)
            # remove members from tags
            self.member_tags = \
                {k: new_members_map[np.setdiff1d(v, members_to_be_deleted)]
                 for k, v in self.member_tags.items()}

    def merge_colinear_nodes(self, epsilon: float = 1e-8) -> None:
        """
        Merge members in co-linear nodes

        See :meth:`tnsgrt.structure.Structure.get_colinear_nodes`

        :param epsilon: accuracy of co-linearity test

        **Notes:**

        1. If the co-linear members are a bar and a string, the node and members are
           skipped
        2. New member inherit all tag from co-linear members
        3. All dependent nodes are removed and co-linear members are replaced
        """
        # get dependent nodes
        dependent_nodes, dependent_members = \
            self.get_colinear_nodes(return_members=True, epsilon=epsilon)
        # loop to create new members
        new_members = np.zeros((2, len(dependent_nodes)), dtype=np.int64)
        new_member_tags = defaultdict(list)
        new_members_to_skip = []
        members_to_be_removed = []
        nodes_to_be_removed = []
        for i, (node, member_indices) in enumerate(zip(dependent_nodes,
                                                       dependent_members)):
            # create member with nodes that are not equal to node
            member_nodes = self.members[:, member_indices]
            new_members[:, i] = member_nodes[member_nodes != node]
            # collect member tags
            tags = set(itertools.chain(*[self.get_member_tags(member)
                                         for member in member_indices]))
            if 'string' in tags and 'bar' in tags:
                new_members_to_skip.append(i)
                warnings.warn(f"Dependent node '{node}' is connected to a bar and "
                              f"a string and cannot be merged")
            else:
                # add node to list of nodes to be removed
                nodes_to_be_removed.append(node)
                # add to members to remove
                members_to_be_removed.extend(member_indices)
                # offset id to match reduced new member numbering
                new_member_id = i - len(new_members_to_skip)
                # add tags to new member
                for tag in tags:
                    new_member_tags[tag].append(new_member_id)
        # trim new_members
        new_members = np.delete(new_members, new_members_to_skip, axis=1)
        if new_members.shape[1] == 0:
            # return if nothing to add
            return
        # make sure member tags are unique
        new_member_tags = \
            {k: np.unique(v).astype(dtype=np.int64)
             for k, v in new_member_tags.items()}
        # remove old members
        self.remove_members(members_to_be_removed)
        # add new members
        self.add_members(members=new_members, member_tags=new_member_tags)
        # remove nodes
        self.remove_nodes(nodes_to_be_removed)

    def merge_overlapping_members(self, verbose: bool = False) -> None:
        """
        Merge overlapping members in structure

        **Notes:**

        1. Overlapping members are members that share the same set of nodes
        2. The properties 'lambda\_', 'force', 'mass', and 'volume' are summed on
           the merged member
        3. The remaining member has as tags the union of all tags in the merged members
        """
        # sort members
        sorted_members = np.sort(self.members, axis=0)
        unique_members, unique_indices, unique_inverse, unique_counts = \
            np.unique(sorted_members, axis=1, return_index=True,
                      return_inverse=True, return_counts=True)

        # select members that appear more than once
        repeated_members = unique_indices[unique_counts > 1]

        # quick return
        if len(repeated_members) == 0:
            if verbose:
                warnings.warn('no members were merged')
            return

        # find out members to be removed and tags to be added
        members_to_be_removed = []
        tags_to_be_added = {}
        for member_idx in unique_inverse[repeated_members]:
            # find repeated members
            repeated = np.nonzero(unique_inverse == member_idx)[0]
            # merge member properties
            self.member_properties.loc[repeated[0],
                                       ['force', 'lambda_', 'mass', 'volume']] = \
                self.member_properties.loc[repeated,
                                           ['force', 'lambda_', 'mass', 'volume']].sum()
            # merge member tags
            existing_tags = set(self.get_member_tags(repeated[0]))
            repeated_tags = \
                set(itertools.chain(*[self.get_member_tags(j) for j in repeated[1:]]))
            new_tags = repeated_tags - existing_tags
            # look for change of character
            is_bar = 'bar' in existing_tags
            if (is_bar and 'string' in new_tags) or (not is_bar and 'bar' in new_tags):
                # check for sign of force coefficient
                lambda_ = self.member_properties.loc[repeated[0], 'lambda_']
                if is_bar and lambda_ > 0:
                    del existing_tags['bar']
                    warnings.warn(f"member {repeated[0]} changed from "
                                  f"'bar' to 'string'")
                elif not is_bar and lambda_ < 0:
                    del existing_tags['string']
                    warnings.warn(f"member {repeated[0]} changed from "
                                  f"'string' to 'bar'")
            tags_to_be_added[repeated[0]] = new_tags
            # which members to remove?
            members_to_be_removed.extend(repeated[1:])

        # invert tag map
        inv_tag_map = collections.defaultdict(list)
        for member, tags in tags_to_be_added.items():
            for tag in tags:
                inv_tag_map[tag].append(member)
        # add tags
        for tag, members in inv_tag_map.items():
            self.add_member_tag(tag, np.array([members]))
        # remove members
        if members_to_be_removed:
            self.remove_members(members_to_be_removed, verbose)

    def get_slack_members(self, epsilon: bool = 1e-8) -> pd.Index:
        """
        :return: the index of members with small force coefficients
        """
        return self.member_properties.index[self.member_properties['lambda_'].abs()
                                            < epsilon]

    def get_number_of_members(self) -> int:
        """
        :return: the number of members in ``Structure``
        """
        return self.members.shape[1]

    def get_member_tags(self, index: int) -> List[str]:
        """
        A list with the tags for the member with index ``index``

        :param index: the index of the member
        :return: list of tags
        """
        return [k for k, v in self.member_tags.items() if index in v]

    def has_member_tag(self, index: int, tag: str) -> bool:
        """
        Return ``True`` if member with index ``index`` has tag ``tag``

        :param index: the index of the member
        :param tag: the tag
        :return: ``True`` or ``False``
        """
        return tag in self.member_tags and index in self.member_tags[tag]

    def get_members_by_tag(self, *tag: str) -> npt.NDArray[np.int64]:
        """
        Return a list of member indices that have given tags

        :param \\*tag: the tag
        :return: list of member indices
        """
        if len(tag) == 0:
            return np.zeros((0,))
        elif len(tag) == 1:
            return self.member_tags.get(tag[0], np.zeros((0,)))
        else:
            return reduce(lambda a1, a2: np.intersect1d(a1, a2, assume_unique=True),
                          [v for k, v in self.member_tags.items() if k in tag])

    def get_number_of_members_by_tag(self, tag: str) -> int:
        """
        Return the number of members with tag ``tag``

        :param tag: the tags
        :return: the number of members
        """
        return len(self.member_tags.get(tag, []))

    def delete_member_tag(self, tag: str) -> None:
        """
        Delete member tag ``tag``

        :param tag: the member tag to be deleted
        """
        # delete member tag
        del self.member_tags[tag]
        if tag in self.member_defaults:
            del self.member_defaults[tag]

    def add_member_tag(self, tag: str,
                       indices: Union[int, npt.NDArray[np.int64]]) -> None:
        """
        Add members with indices in ``indices`` to the member tag ``tag``

        Create tag if it does not already exist

        :param tag: the member tag
        :param indices: the member indices
        """

        # put int in a list
        if isinstance(indices, int):
            indices = [indices]

        # quick return
        if len(indices) == 0:
            return

        # make sure indices are valid
        number_of_members = self.get_number_of_members()
        assert np.amin(indices) >= 0, \
            'member tag index must be greater or equal than zero'
        assert np.amax(indices) < number_of_members, \
            'member tag index must be less than number of members'

        if tag in self.member_tags:
            # set tag
            self.member_tags[tag] = np.union1d(self.member_tags[tag], indices)
        else:
            # add tag
            self.member_tags[tag] = np.unique(indices)

    def remove_member_tag(self, tag: str, indices: npt.NDArray[np.int64]) -> None:
        """
        Remove members with indices in ``indices`` from the existing member tag ``tag``

        :param tag: the member tag
        :param indices: the member indices
        """
        # remove indices from existing member tag

        # set tag
        self.member_tags[tag] = np.setdiff1d(self.member_tags[tag], indices)

    def delete_node_tag(self, tag: str) -> None:
        """
        Delete node tag ``tag``

        :param tag: the node tag to be deleted
        """
        del self.node_tags[tag]
        if tag in self.node_defaults:
            del self.node_defaults[tag]

    def add_node_tag(self, tag: str, indices: npt.NDArray[np.int64]) -> None:
        """
        Add nodes with indices in ``indices`` to the node tag ``tag``

        Create tag if it does not already exist

        :param tag: the node tag
        :param indices: the node indices
        """

        # put int in a list
        if isinstance(indices, int):
            indices = [indices]

        # quick return
        if len(indices) == 0:
            return

        # make sure indices are valid
        assert np.amin(indices) >= 0, \
            'node tag index must be greater or equal than zero'
        assert np.amax(indices) < self.get_number_of_nodes(), \
            'node tag index must be less than number of nodes'

        if tag in self.node_tags:
            # set tag
            self.node_tags[tag] = np.union1d(self.node_tags[tag], indices)
        else:
            # create tag
            self.node_tags[tag] = np.unique(indices)

    def remove_node_tag(self, tag: str, indices: npt.NDArray[np.int64]) -> None:
        """
        Remove nodes with indices in ``indices`` from the existing node tag ``tag``

        :param tag: the node tag
        :param indices: the node indices
        """

        # set tag
        self.node_tags[tag] = np.setdiff1d(self.node_tags[tag], indices)

    @staticmethod
    def _set_dataframe(df: pd.DataFrame,
                       index: Union[int, Sequence[int], slice],
                       labels: Union[str, Sequence[str]],
                       values: Any, scalar: bool = True) -> None:
        """
        Auxiliary set method to wrap values when setting dataframe

        :param df: the dataframe
        :param index: the row index
        :param labels: the column label(s)
        :param values: the values to set to
        :param scalar: if ``True``, ``value`` is set to an array of len(index)

        **Notes:**

        1. This method is for convenience when assigning objects to dataframes.
           See `this question <https://stackoverflow.com/questions/48000225/\
must-have-equal-len-keys-and-value-when-setting-with-an-iterable>`_
           for details.
        """
        if isinstance(index, int):
            idx = index = [index]
            m = 1
        else:
            idx = df.index[index]
            m = len(idx)
        if scalar:
            values = [values] * m
        if isinstance(labels, str):
            values = pd.Series(values, index=idx)
        else:
            values = pd.DataFrame(values, index=idx)
        df.loc[index, labels] = values

    def set_member_properties(self, index: Union[int, Sequence[int], slice],
                              labels: Union[str, Sequence[str]], values: Any, *vargs,
                              scalar: bool = True) -> None:
        """
        Set member properties

        See :meth:`tnsgrt.structure.Structure._set_dataframe`

        :param index: the element index
        :param labels: the property label(s)
        :param values: the values to set to
        :param \\*vargs: label/values pairs
        :param scalar: if ``True``, ``value`` is set to an array of len(index)
        """
        Structure._set_dataframe(self.member_properties,
                                 index, labels, values, scalar=scalar)
        for lv in more_itertools.batched(vargs, 2):
            Structure._set_dataframe(self.member_properties, index, *lv, scalar=scalar)

    def get_member_properties(self,
                              index: Union[int, Sequence[int], slice], *labels: str)\
            -> pd.DataFrame:
        """
        Retrieve member properties

        :param index: the member index
        :param \\*labels: the member property labels
        :return: datafrome with the selected properties

        **WARNING:** :meth:`tnsgrt.structure.Structure.get_member_properties` uses
        pandas' `loc` method that includes the last element of slices; See
        `pandas documentation <https://pandas.pydata.org/docs/reference/api/\
pandas.DataFrame.loc.html>`_
        for details
        """
        return \
            self.member_properties.loc[index,
                                       labels[0] if len(labels) == 1 else list(labels)]

    def set_node_properties(self, index: Union[int, Sequence[int], slice],
                            labels: Union[str, Sequence[str]], values: Any, *vargs,
                            scalar: bool = True) -> None:
        """
        Set node properties

        See :meth:`tnsgrt.structure.Structure._set_dataframe`

        :param index: the element index
        :param labels: the property label(s)
        :param values: the values to set to
        :param \\*vargs: label/values pairs
        :param scalar: if ``True``, ``value`` is set to an array of len(index)
        """
        Structure._set_dataframe(self.node_properties, index, labels, values, scalar)
        for lv in more_itertools.batched(vargs, 2):
            Structure._set_dataframe(self.node_properties, index, *lv, scalar=scalar)

    def get_node_properties(self,
                            index: Union[int, Sequence[int], slice], *labels: str)\
            -> pd.DataFrame:
        """
        Retrieve node properties

        :param index: the node index
        :param \\*labels: the node property labels
        :return: datafrome with the selected properties

        **WARNING:** :meth:`tnsgrt.structure.Structure.get_node_properties` uses
        pandas' `loc` method that includes the last element of slices; See
        `pandas documentation <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html>`_
        for details
        """
        return self.node_properties.loc[index,
                                        labels[0] if len(labels) == 1 else list(labels)]

    def get_member_vectors(self) -> npt.NDArray[np.float_]:
        """
        :return: a 3 x m array with the ``Structure``'s members
        """
        return self.nodes[:, self.members[1, :]] - self.nodes[:, self.members[0, :]]

    def get_member_length(self) -> npt.NDArray[np.float_]:
        """
        :return: an array with the ``Structure``'s member lengths
        """
        return np.linalg.norm(self.nodes[:, self.members[1, :]] - \
                              self.nodes[:, self.members[0, :]], axis=0)

    def get_center_of_mass(self) -> npt.NDArray[np.float_]:
        """
        :return: the ``Structure``'s center of mass
        """
        # Computes the center of mass
        mass = self.member_properties['mass'].values
        return np.sum(mass * (self.nodes[:, self.members[0, :]] +
                              self.nodes[:, self.members[1, :]])/2, axis=1) \
            / np.sum(mass)

    def get_centroid(self) -> npt.NDArray[np.float_]:
        """
        :return: the ``Structure``'s geometric center or centroid
        """
        # Computes the centroid (geometric center)
        return np.sum(self.nodes, axis=1) / self.get_number_of_nodes()

    def update_member_properties(self,
                                 property_name: Optional[Union[str,
                                                         Iterable[str]]] = None) \
            -> None:
        """
        Update ``Structure``'s member properties

        If property_name is

        - 'stiffness': calculate ``stiffness`` and ``rest_length`` based on
          ``modulus``, ``radius``, ``inner_radius`` and ``lambda_``
        - 'mass': calculate ``mass`` and ``volume`` based on ``radius`` and ``density``
        - 'force': calculate ``force`` based on ``lambda_``

        :param property_name: the property name to update; if ``None``, update all
                              properties
        """

        if isinstance(property_name, str):

            # update stiffness and rest length
            if property_name == 'stiffness':
                member_length = self.get_member_length()
                modulus_times_area = \
                    self.member_properties['modulus'].values * np.pi * \
                    (self.member_properties['radius'].values ** 2 -
                     self.member_properties['inner_radius'].values ** 2)
                stiffness = modulus_times_area / member_length
                if np.any(stiffness < 0):
                    raise Exception('Structure::update_member_property: '
                                    'negative stiffness computed')
                rest_length = \
                    member_length * \
                    (1 - self.member_properties['lambda_'].values / stiffness)
                if np.any(rest_length < 0):
                    raise Exception('Structure::update_member_property: '
                                    'negative rest_length computed')
                self.member_properties['stiffness'] = stiffness
                self.member_properties['rest_length'] = rest_length

            # update mass and volume
            elif property_name == 'mass':
                member_length = self.get_member_length()
                volume = \
                    np.pi * (self.member_properties['radius'] ** 2 -
                             self.member_properties['inner_radius'].values ** 2) * \
                    member_length
                if np.any(volume < 0):
                    raise Exception('Structure::update_member_property: '
                                    'negative volume computed')
                mass = volume * self.member_properties['density']
                if np.any(mass < 0):
                    raise Exception('Structure::update_member_property: '
                                    'negative mass computed')
                self.member_properties['volume'] = volume
                self.member_properties['mass'] = mass

            # update force
            elif property_name == 'force':
                self.member_properties['force'] = \
                    self.get_member_length() * self.member_properties['lambda_'].values

            else:
                raise Exception('Structure::update_member_property: '
                                f'do not know how to update property {property_name}')

        else:

            # iterate all if None
            if property_name is None:
                property_name = ['mass', 'stiffness', 'force']

            # iterate properties
            for prop in property_name:
                self.update_member_properties(prop)

    def get_close_nodes(self, radius=1e-6) -> Tuple[Set[int], npt.NDArray]:
        """
        Returns the set of nodes that lie within ``radius`` distance to other nodes in
        ``Structure`` and the corresponding map

        For example, if::

            close_nodes, close_nodes_map = s.get_close_nodes()

        is such that::

            close_nodes = {1, 3}
            close_nodes_map = [0,0,2,2,4]

        then node ``1`` is close to node ``0`` and node ``3`` is close to node ``2``.

        :param radius: the proximity radius
        :return: tuple with ``close_nodes`` and ``close_nodes_map``
        """

        tree = scipy.spatial.KDTree(self.nodes.transpose())
        indices = tree.query_ball_tree(tree, r=radius)
        close_nodes_map = np.arange(self.get_number_of_nodes())
        close_nodes_set = set()
        for i, neighbors in enumerate(indices):
            for k in [j for j in neighbors if j > i]:
                close_nodes_map[k] = i
                close_nodes_set.add(k)
        if close_nodes_set:
            # apply map until no changes
            while True:
                last_merge_map = close_nodes_map.copy()
                close_nodes_map = close_nodes_map[close_nodes_map]
                if np.all(last_merge_map == close_nodes_map):
                    break
        return close_nodes_set, close_nodes_map

    def merge_close_nodes(self, radius: float = 1e-6, verbose: bool = False) -> None:
        """
        Merge then remove all nodes in ``Structure`` which lie within ``radius``

        :param radius: the proximity radius
        :param verbose: if ``True`` issues a warning with number of nodes removed after
                        the merge
        """
        # get close nodes
        close_nodes_set, close_nodes_map = self.get_close_nodes(radius)
        if close_nodes_set:
            # list of nodes to be removed
            nodes_to_be_removed = list(close_nodes_set)
            # apply merge_map to members
            self.members = close_nodes_map[self.members]
            # remove nodes, to make sure the member node numbering is correct
            self.remove_nodes(nodes_to_be_removed,
                              verify_if_unused=False, verbose=verbose)

    def merge(self, s: 'Structure') -> None:
        """
        Merge ``s`` with the current structure

        :param s: the ``Structure`` to merge
        """

        # offset members in s by number_of_members
        number_of_members = self.get_number_of_members()
        number_of_nodes = self.get_number_of_nodes()

        # merge nodes
        self.nodes = np.hstack((self.nodes, s.nodes))

        # merge node tags
        for k, v in s.node_tags.items():
            # append or add
            self.node_tags[k] = np.hstack((self.node_tags[k], number_of_nodes + v)) \
                if k in self.node_tags \
                else number_of_nodes + v

        # merge node properties
        self.node_properties = pd.concat((self.node_properties, s.node_properties),
                                         ignore_index=True)

        # merge members, offset by number of nodes
        self.members = np.hstack((self.members, number_of_nodes + s.members))

        # merge member tags
        for k, v in s.member_tags.items():
            # append or add
            self.member_tags[k] = np.hstack((self.member_tags[k],
                                             number_of_members + v)) \
                if k in self.member_tags \
                else number_of_members + v

        # merge member properties
        self.member_properties = \
            pd.concat((self.member_properties, s.member_properties), ignore_index=True)

    def equilibrium(self,
                    force: Optional[npt.ArrayLike] = None,
                    lambda_bar: Optional[float] = None,
                    equalities: Optional[List[npt.ArrayLike]] = None,
                    epsilon: float = 1e-7) -> None:
        """
        Solves for the set of internal forces that ensures the equilibrium of the
        current ``Structure`` in response to the vector of external forces ``forces``

        Solve the force equilibrium equation

        .. math::
           A \\lambda = f, \\quad \\lambda \\in \\Lambda

        in which:

        - :math:`A`: is a matrix representing the element vectors
        - :math:`f`: is the vector of external forces
        - :math:`\\lambda`: is the vector of force coefficients
        - :math:`\\Lambda`: is a set of constraints on the force coefficients
                            (see Notes below)

        :param force: a 3 x n array of external forces or ``None``
        :param lambda_bar: the normalizing factor
        :param equalities: a list of lists of member indices which are constrained to
                           have the same force coefficient
        :param epsilon: numerical accuracy

        **Notes:**

        1. If the :math:`i` th element is a string then

            .. math::
                \\Lambda = \\{ \\lambda : \\qquad \\lambda_i \\geq 0 \\quad
                \\text{ if } i\\text{th element is a string} \\}

        2. All elements in ``equalities`` are set equal. For example, if::

               equalities = [[0, 1, 3], [2, 4]]

           then the constraints

            .. math::
                \\lambda_0 = \\lambda_1 = \\lambda_3, \\qquad \\lambda_2 = \\lambda_4

           are added to the constraint set :math:`\\Lambda`.

        3. If ``force=None`` then the sum of the bar force coefficients equals
           ``lambda_bar``. That is, the following modified problem is solved:

            .. math::
               A \\lambda = 0, \\quad \\mathbf{e}^T \\lambda = \\bar{\\lambda},
               \\quad \\lambda \\in \\Lambda

           in which :math:`\\mathbf{e}` is a vector that has `1` for bars and `0`
           for strings.

        4. If ``force`` is not ``None`` and ``lambda_bar`` is also not ``None`` then
           the following problem is solved

            .. math::
                A \\lambda = f, \\quad \\mathbf{e}^T \\lambda = \\bar{\\lambda},
                \\quad \\lambda \\in \\Lambda

            **WARNING:** This problem may not be feasible for all
            :math:`\\bar{\\lambda} > 0`!

        """

        number_of_nodes = self.get_number_of_nodes()
        number_of_strings = len(self.member_tags.get('string', []))
        number_of_bars = len(self.member_tags.get('bar', []))
        number_of_members = number_of_strings + number_of_bars

        assert number_of_members == number_of_bars + number_of_strings, \
            'number of members is not equal to the sum of number of bars and strings'

        assert lambda_bar is None or lambda_bar >= 0, 'lambda_bar cannot be negative'

        # member vectors
        member_vectors = self.get_member_vectors()
        strings = self.member_tags['string']

        # coefficient matrix
        Aeq = np.zeros((3 * number_of_nodes, number_of_members))

        # string coefficient
        if number_of_strings:
            for i in self.member_tags['string']:

                ii = 3 * int(self.members[0, i])
                Aeq[ii:ii+3, i] = -member_vectors[:, i]

                jj = 3 * int(self.members[1, i])
                Aeq[jj:jj+3, i] = member_vectors[:, i]

        # bar coefficient
        if number_of_bars:
            for i in self.member_tags['bar']:

                ii = 3 * int(self.members[0, i])
                Aeq[ii:ii+3, i] = member_vectors[:, i]

                jj = 3 * int(self.members[1, i])
                Aeq[jj:jj+3, i] = -member_vectors[:, i]

        A = Aeq
        m = 3 * number_of_nodes

        # sum of the bars
        ee = np.ones((1, number_of_members)) / self.get_number_of_members_by_tag('bar')
        ee[:, strings] = 0
        # TODO: do we need to worry if no bars?

        if force is None:
            # no external forces
            # A = np.vstack((A, np.ones((1, number_of_members))))
            A = np.vstack((A, ee))
            blo = bup = np.hstack((np.zeros((3 * number_of_nodes,)), 1))
            if lambda_bar is None:
                lambda_bar = 1
        else:
            # external forces
            beq = np.array(force)
            assert np.all(beq.shape == (3, number_of_nodes)), \
                'force must be a 3 x n matrix'
            blo = bup = beq.flatten(order='F')
            if lambda_bar is not None:
                # pretension under external forces
                A = np.vstack((A, ee))
                blo = np.hstack((blo, lambda_bar))
                bup = np.hstack((bup, lambda_bar))

        # For equilibrium: Aeq x = beq

        # impose equalities
        # indices in each row are set to be equal
        if equalities is not None:
            number_of_constraints = sum(map(len, equalities)) - len(equalities)
            Aeq = np.zeros((number_of_constraints, number_of_members))
            beq = np.zeros((number_of_constraints, ))
            ii = 0
            for eqs in equalities:
                nn = len(eqs)
                for jj in range(1, nn):
                    Aeq[ii+jj-1, eqs[0]] = 1
                    Aeq[ii+jj-1, eqs[jj]] = -1
                ii += nn
            A = np.vstack((A, Aeq))
            blo = bup = np.hstack((blo, beq))

        # enforce strings have positive force coefficients
        # when there are no external forces set to one to avoid nontrivial solution
        xup = None
        xlo = None
        if number_of_strings:
            xlo = np.full((number_of_members, ), -np.inf)
            xlo[self.member_tags['string']] = 0

        # cost function
        n = number_of_members
        c = np.ones((n,))

        # solve lp
        cost, gamma, status = optim.lp(n, m, c, A, blo, bup, xlo, xup)

        # if infeasible, throw error
        if status == 'infeasible':
            raise Exception('could not find equilibrium')

        # flip sign for bars
        lambda_ = gamma
        if number_of_bars:
            lambda_[self.member_tags['bar']] *= -1

            if force is None:
                # scale solution for bars
                scale = -np.sum(lambda_[self.member_tags['bar']]) / number_of_bars
                lambda_ *= np.abs(lambda_bar) / scale

        # assign lambda
        self.member_properties['lambda_'] = lambda_

        # update force
        self.update_member_properties('force')

    def stiffness(self, epsilon: float = 1e-6, storage: str = 'sparse',
                  apply_rigid_body_constraint: bool = False,
                  apply_planar_constraint: bool = False):
        """
        Computes

        - `normal`: potential energy (1 x 1)
        - `F`: force vectors (3 x n)
        - `K`: stiffness matrix (3 n x 3 n)
        - `M`: mass matrix (n x 1)

        for the current ``Structure``. The mass and stiffness matrices are returned in
        the form of an object of the class :class:`tnsgrt.stiffness.Stiffness`

        :param epsilon: numerical accuracy
        :param storage: if ``sparse`` stores the resulting stiffness and mass matrices
                        in sparse csr format
        :param apply_rigid_body_constraint: if ``True`` apply 3D rigid body constraints
        :param apply_planar_constraint: if ``True`` apply 2D constraints
        :return: tuple (`S`, `F`, `normal`)
        """

        number_of_nodes = self.get_number_of_nodes()
        number_of_strings = len(self.member_tags.get('string', []))
        number_of_bars = len(self.member_tags.get('bar', []))
        number_of_members = number_of_strings + number_of_bars

        assert number_of_members == number_of_bars + number_of_strings, \
            'number of members is not equal to the sum of number of bars and strings'

        if number_of_nodes <= 12 and storage == 'sparse':
            storage = 'dense'
            warnings.warn("number of nodes is small; storage set to 'dense'")

        # member vectors
        member_vectors = self.get_member_vectors()
        member_length = self.get_member_length()

        k = self.member_properties['stiffness'].values
        lambda_ = self.member_properties['lambda_'].values
        mass = self.member_properties['mass'].values

        # compute potential
        v = np.sum(((lambda_ * member_length) ** 2) / k / 2)

        # Compute force and stiffness

        # preallocate F
        F = np.zeros((3, number_of_nodes))
        # preallocate K
        if storage == 'sparse':
            # sparse storage
            diag = np.unique(self.members)
            diff = np.unique(self.members, axis=1)
            row_col = np.hstack((np.vstack((diag, diag)), diff, np.flipud(diff)))
            data = np.zeros((row_col.shape[1]))
            K = scipy.sparse.kron(
                scipy.sparse.coo_matrix((data, row_col),
                                        shape=(number_of_nodes, number_of_nodes)),
                np.ones((3, 3))).tocsr()
        else:
            # dense storage
            K = np.zeros((3 * number_of_nodes, 3 * number_of_nodes))

        lambda_ * member_vectors
        # calculate stiffness and force
        for i in range(number_of_members):

            mij = member_vectors[:, i] / member_length[i]
            fij = lambda_[i] * member_vectors[:, i]

            mijmij = np.outer(mij, mij)
            # kij = k[index] * Mij + lambda_[index] * (np.eye(3) - Mij)
            kij = (k[i] - lambda_[i]) * mijmij + lambda_[i] * np.eye(3)

            i, j = self.members[:, i].astype(dtype=np.int_)

            F[:, i] += fij
            F[:, j] -= fij

            ii = 3 * i
            jj = 3 * j

            K[ii:ii+3, ii:ii+3] += kij
            K[jj:jj+3, jj:jj+3] += kij
            K[ii:ii+3, jj:jj+3] -= kij
            K[jj:jj+3, ii:ii+3] -= kij

        # Compute mass
        M = np.zeros((number_of_nodes,))
        M[self.members[0, :]] = mass / 2
        M[self.members[1, :]] += mass / 2

        diag = np.kron(M, np.ones((3,)))
        if storage == 'sparse':
            # sparse storage
            M = scipy.sparse.diags(diag, format='csr')
        else:
            # dense storage
            M = np.diag(np.kron(M, np.ones((3,))))

        # Check forces
        sum_of_forces = np.linalg.norm(F, ord='fro')
        if sum_of_forces > epsilon:
            warnings.warn(f'Structure::stiffness: force balance not satisfied, '
                          f'sum of forces = {sum_of_forces}')

        # Build stiffness object
        stiffness = Stiffness(K, M)

        # node constraints
        constraints = self.node_properties['constraint']
        has_constraints = constraints.isna().sum() < self.get_number_of_nodes()
        if has_constraints:
            # apply node constraints
            stiffness.apply_constraint(
                *NodeConstraint.node_constraint(self.nodes, constraints))
        if apply_rigid_body_constraint:
            # apply rigid body constraints
            stiffness.apply_constraint(
                *NodeConstraint.rigid_body_constraint(self.nodes), local=False)
        if apply_planar_constraint:
            # apply planar constraints
            stiffness.apply_constraint(
                *NodeConstraint.planar_constraint(self.nodes), local=False)

        return stiffness, F, v


# module methods
def rotate(s: Structure, v: npt.NDArray) -> Structure:
    """
    Returns a copy of ``s`` in which all nodes are rotated around the 3D vector ``v``

    :param s: the structure to rotate
    :param v: the 3D rotation vector
    :return: the rotated structure

    **Notes:**

    1. See :meth:`scipy.spatial.transform.Rotation.from_rotvec` for details
    """
    return s.copy().rotate(v)


def translate(s: Structure, v: npt.NDArray) -> Structure:
    """
    Returns a copy of ``s`` in which all nodes are translated by the 3D vector ``v``

    :param s: the structure to translate
    :param v: the 3D translation vector
    :return: the translated structure
    """
    return s.copy().translate(v)


def reflect(s: Structure, v: npt.NDArray, p: npt.NDArray) -> Structure:
    """
    Returns a copy of ``s`` in which all nodes are reflected about a plane normal to
    the vector `v`, passing through the point `p`.

    If no point is given, it defaults to the origin.

    :param s: the structure to reflect
    :param v: the 3D normal vector
    :param p: the 3D origin vector
    :return: the reflected structure
    """
    return s.copy().reflect(v, p)


def merge(*s: Structure) -> Structure:
    """
    Returns a new structure in which all structures given in `s` are merged

    :param \\*s: the structures to merge
    :return: the merged structure
    """
    if not s:
        return Structure()
    else:
        r = s[0].copy()
        for si in s[1:]:
            r.merge(si)
        return r
