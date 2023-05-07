import warnings
from dataclasses import dataclass
from functools import reduce
from typing import Optional, Dict, get_type_hints, Union, List, Sequence, Type, Iterable, Tuple, Set
from collections import ChainMap

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy

from tensegrity import optim


@dataclass
class Property:

    @classmethod
    def to_dataframe(cls: Type['Property'], data: Union[list, tuple] = tuple()) -> pd.DataFrame:
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
        facecolor: object = (0, 0.4470, 0.7410)
        edgecolor: object = (0, 0.4470, 0.7410)

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
            'facecolor': (0, 0.4470, 0.7410),
            'edgecolor': (0, 0.4470, 0.7410)
        },
        'string': {
            'facecolor': (0.8500, 0.3250, 0.0980),
            'edgecolor': (0.8500, 0.3250, 0.0980)
        }
    }

    def __init__(self,
                 nodes: npt.ArrayLike = np.zeros((3, 0), np.float_),
                 members: npt.ArrayLike = np.zeros((2, 0), np.int64),
                 number_of_strings: int = 0,
                 node_tags: Optional[Dict[str, npt.NDArray[np.uint64]]] = None,
                 member_tags: Optional[Dict[str, npt.NDArray[np.uint64]]] = None,
                 label: str = None):
        # label
        self.label: Optional[str] = label
        # nodes
        self.nodes: npt.NDArray[np.float_] = np.zeros((3, 0), np.float_)
        self.node_tags: Dict[str, npt.NDArray[np.uint64]] = {}
        self.node_properties: pd.DataFrame = Structure.NodeProperty.to_dataframe()
        # members
        self.members: npt.NDArray[np.uint64] = np.zeros((2, 0), np.uint64)
        self.member_tags: Dict[str, npt.NDArray[np.uint64]] = {
            'bar': np.zeros((0,), np.uint64),
            'string': np.zeros((0,), np.uint64)
        }
        self.member_properties: pd.DataFrame = Structure.MemberProperty.to_dataframe()

        # add nodes
        self.add_nodes(nodes, node_tags)

        # add members
        self.add_members(members, number_of_strings, member_tags)

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
        assert np.all(nodes.shape == self.nodes.shape), 'nodes shape must match current shape'

        # set nodes
        self.nodes: npt.NDArray[np.float_] = nodes

    def add_nodes(self, nodes: npt.ArrayLike,
                  node_tags: Optional[Dict[str, npt.NDArray[np.uint64]]] = None) -> None:
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
        tags_with_defaults = list(set(node_tags.keys()) & set(Structure.node_defaults.keys()))
        # apply defaults
        new_node_properties = [Structure.NodeProperty(**ChainMap(*[Structure.node_defaults[tag]
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
        self.node_properties = pd.concat((self.node_properties,
                                          Structure.NodeProperty.to_dataframe(new_node_properties)),
                                         ignore_index=True)

    def get_number_of_nodes(self) -> int:
        """
        :return: the number of nodes in ``Structure``
        """
        return self.nodes.shape[1]

    def translate(self, v: npt.NDArray) -> 'Structure':
        """
        Translate all nodes of the ``Structure`` by the 3D vector ``v``

        :param v: the 3D translation vector
        :return: self
        """
        # `translate(v)` translates nodes of the structure by the 3D vector `v`
        assert v.shape == (3,), 'v must be a three dimensional vector'
        self.nodes += v.reshape((3, 1))
        return self

    def rotate(self, v: npt.NDArray) -> 'Structure':
        """
        Rotate all nodes of the ``Structure`` by the 3D vector ``v``

        :param v: the 3D rotation vector
        :return: self
        """
        # `rotate(v)` rotates the nodes of the structure around the 3D vector `v`
        # the magnitude of the vector equals the rotation angle in radians
        assert v.shape == (3,), 'v must be a three dimensional vector'
        rotation = scipy.spatial.transform.Rotation.from_rotvec(v)
        self.nodes = rotation.apply(self.nodes.transpose()).transpose()
        return self

    def reflect(self, v: npt.NDArray, p: Optional[npt.NDArray] = None) -> 'Structure':
        """
        Reflects the structure about a plane normal to the vector `v`, passing through the point `p`.
        If no point is given, it defaults to the origin.

        :param v: the 3D normal vector
        :param p: the 3D origin vector
        :return: self
        """
        assert v.shape == (3,), 'v must be a three dimensional vector'

        if p is not None:
            assert p.shape == (3,), 'p must be a three dimensional vector'
            # translate by p
            self.nodes -= p.reshape((3, 1))

        # normalize v
        length = np.linalg.norm(v)
        if length < 1e-6:
            warnings.warn('norm of vector v is too small, reflection not performed')
            return self

        # calculate reflection matrix
        reflection_matrix = np.eye(3) - (2 / length**2) * np.outer(v, v)

        # transform nodes
        self.nodes = reflection_matrix @ self.nodes

        if p is not None:
            # translate back to p
            self.nodes += p.reshape((3, 1))

        return self

    def get_unused_nodes(self) -> npt.NDArray[np.uint64]:
        """
        :return: an array with the indices of the unused nodes
        """
        # calculate nodes that are in use
        used_nodes = np.unique(self.members)
        # return unused nodes
        return np.setdiff1d(np.arange(self.get_number_of_nodes()), used_nodes, assume_unique=True)

    def has_unused_nodes(self) -> bool:
        """
        :return: ``True`` if there are no unused nodes
        """
        return len(self.get_unused_nodes()) > 0

    def remove_nodes(self, nodes_to_be_deleted: Optional[npt.ArrayLike] = None,
                     verify_if_unused: bool = True, verbose: bool = False) -> None:
        """
        Remove nodes from structure

        :param nodes_to_be_deleted: the indices of the nodes to be deleted; if ``None``,
                                    delete all currently unused nodes
        :param verify_if_unused: if ``True`` verifies if the nodes to be deleted are not in use
        :param verbose: if ``True`` warns of the nodes to be deleted
        """
        if nodes_to_be_deleted is None:
            # delete all unused nodes
            unused_nodes_to_be_deleted = self.get_unused_nodes()
        elif verify_if_unused:
            # sort nodes to be deleted
            nodes_to_be_deleted = np.unique(nodes_to_be_deleted)
            # calculate nodes that are in use
            used_nodes = np.unique(self.members)
            # find unused nodes
            unused_nodes_to_be_deleted = np.setdiff1d(nodes_to_be_deleted, used_nodes, assume_unique=True)
            # warn if different
            number_of_used_nodes = len(nodes_to_be_deleted) - len(unused_nodes_to_be_deleted)
            if number_of_used_nodes:
                warnings.warn(f'{number_of_used_nodes} nodes are still in use and were not deleted')
                if verbose:
                    warnings.warn('The following nodes will not be removed: ' 
                                  f'{np.intersect1d(nodes_to_be_deleted, used_nodes, assume_unique=True)}')
        else:
            # go ahead without verifying if nodes are unused
            # WARNING: this may result in orphan members!
            unused_nodes_to_be_deleted = nodes_to_be_deleted
        # delete if there are any unused nodes
        if len(unused_nodes_to_be_deleted):
            if verbose:
                warnings.warn('The following nodes will be removed: '
                              f'{unused_nodes_to_be_deleted}')
            # create new node map
            node_index = np.delete(np.arange(self.get_number_of_nodes()), unused_nodes_to_be_deleted)
            new_node_map = np.zeros((self.get_number_of_nodes(),), dtype=np.int_)
            new_node_map[node_index] = np.arange(self.get_number_of_nodes() - len(unused_nodes_to_be_deleted))
            # remove nodes
            self.nodes = np.delete(self.nodes, unused_nodes_to_be_deleted, axis=1)
            # remove node properties
            self.node_properties.drop(unused_nodes_to_be_deleted, inplace=True)
            self.node_properties.reset_index(inplace=True)
            # remove nodes from tags
            self.node_tags = {k: new_node_map[np.setdiff1d(v, unused_nodes_to_be_deleted)]
                              for k, v in self.node_tags.items()}
            # apply new node map to members
            self.members = new_node_map[self.members]

    def add_members(self, members: npt.ArrayLike,
                    number_of_strings: Optional[int] = None,
                    member_tags: Optional[Dict[str, npt.NDArray[np.uint64]]] = None) -> None:
        """
        Add members and tags to current structure

        :param members: the members to be added
        :param number_of_strings: the number of strings; if not ``None``,  then the first `number_of_strings` members
                                  are tagged as 'strings' and the remaining members as 'bars'
        :param  member_tags: the new members' tags
        """

        # convert to array
        members = np.array(members, np.uint64)

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
                'string': np.arange(0, number_of_strings, dtype=np.uint64),
                'bar': number_of_strings + np.arange(0, number_of_new_bars, dtype=np.uint64)
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
                   np.intersect1d(member_tags['bar'], member_tags['string'], assume_unique=True).size == 0, \
                'bar and string tags must be mutually exclusive'
            # update member_tags
            new_member_tags.update(member_tags)

        # new member properties
        number_of_members = self.get_number_of_members()
        # determine tags that have defaults
        tags_with_defaults = list(set(new_member_tags.keys()) & set(Structure.member_defaults.keys()))
        # apply defaults
        new_member_properties = [Structure.MemberProperty(**ChainMap(*[Structure.member_defaults[tag]
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
            self.member_tags[k] = np.hstack((self.member_tags[k], number_of_members + v)) \
                if k in self.member_tags else number_of_members + v

        # add default member properties
        self.member_properties = pd.concat((self.member_properties,
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
        if members_to_be_deleted:
            if verbose:
                warnings.warn('The following members will be removed: '
                              f'{members_to_be_deleted}')
            # create new member map
            member_index = np.delete(np.arange(self.get_number_of_members()), members_to_be_deleted)
            new_members_map = np.zeros((self.get_number_of_members(),), dtype=np.int_)
            new_members_map[member_index] = np.arange(self.get_number_of_members() - len(members_to_be_deleted))
            # remove members
            self.members = np.delete(self.members, members_to_be_deleted, axis=1)
            # remove member properties
            self.member_properties.drop(members_to_be_deleted, inplace=True)
            self.member_properties.reset_index(inplace=True)
            # remove members from tags
            self.member_tags = {k: new_members_map[np.setdiff1d(v, members_to_be_deleted)]
                                 for k, v in self.member_tags.items()}

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

    def get_members_by_tags(self, tags: Sequence[str]) -> npt.NDArray[np.uint64]:
        """
        Return a list of member indices that have tags in the sequence ``tags``

        :param tags: the sequence of tags
        :return: list of member indices
        """
        if len(tags) == 0:
            return np.zeros((0,))
        elif len(tags) == 1:
            return self.member_tags.get(tags[0], np.zeros((0,)))
        else:
            return reduce(lambda a1, a2: np.intersect1d(a1, a2, assume_unique=True),
                          [v for k, v in self.member_tags.items() if k in tags])

    def delete_member_tag(self, tag: str) -> None:
        """
        Delete member tag ``tag``

        :param tag: the member tag to be deleted
        """
        # delete member tag
        del self.member_tags[tag]
        if tag in self.member_defaults:
            del self.member_defaults[tag]

    def set_member_tag(self, tag: str, indices: npt.NDArray[np.uint64]) -> None:
        """
        Associate members with indices in ``indices`` to the new tag ``tag``

        :param tag: the member tag
        :param indices: the member indices
        """
        # set new member tag

        # make sure tag is unique
        assert tag not in self.member_tags, f"member tag '{tag}' already exists"

        # normalize indices
        v = np.unique(indices)

        # make sure indices are valid
        number_of_members = self.get_number_of_members()
        assert np.amin(v) >= 0, \
            'member tag index must be greater or equal than zero'
        assert np.amax(v) < number_of_members, \
            'member tag index must be less than number of members'

        # set tag
        self.member_tags[tag] = v

    def add_member_tag(self, tag: str, indices: npt.NDArray[np.uint64]) -> None:
        """
        Add members with indices in ``indices`` to the existing member tag ``tag``

        :param tag: the member tag
        :param indices: the member indices
        """

        # make sure indices are valid
        number_of_members = self.get_number_of_members()
        assert np.amin(indices) >= 0, \
            'member tag index must be greater or equal than zero'
        assert np.amax(indices) < number_of_members, \
            'member tag index must be less than number of members'

        # set tag
        self.member_tags[tag] = np.union1d(self.member_tags[tag], indices)

    def remove_member_tag(self, tag: str, indices: npt.NDArray[np.uint64]) -> None:
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

    def set_node_tag(self, tag: str, indices: npt.NDArray[np.uint64]) -> None:
        """
        Associate nodes with indices in ``indices`` to the new tag ``tag``

        :param tag: the node tag
        :param indices: the node indices
        """

        # make sure tag is unique
        assert tag not in self.node_tags, f"node tag '{tag}' already exists"

        # normalize indices
        v = np.unique(indices)

        # make sure indices are valid
        assert np.amin(v) >= 0, \
            'node tag index must be greater or equal than zero'
        assert np.amax(v) < self.get_number_of_nodes(), \
            'node tag index must be less than number of nodes'

        # set tag
        self.node_tags[tag] = v

    def add_node_tag(self, tag: str, indices: npt.NDArray[np.uint64]) -> None:
        """
        Add nodes with indices in ``indices`` to the existing node tag ``tag``

        :param tag: the node tag
        :param indices: the node indices
        """

        # make sure indices are valid
        assert np.amin(indices) >= 0, \
            'node tag index must be greater or equal than zero'
        assert np.amax(indices) < self.get_number_of_nodes(), \
            'node tag index must be less than number of nodes'

        # set tag
        self.node_tags[tag] = np.union1d(self.node_tags[tag], indices)

    def remove_node_tag(self, tag: str, indices: npt.NDArray[np.uint64]) -> None:
        """
        Remove nodes with indices in ``indices`` from the existing node tag ``tag``

        :param tag: the node tag
        :param indices: the node indices
        """

        # set tag
        self.node_tags[tag] = np.setdiff1d(self.node_tags[tag], indices)

    def get_member_properties(self, index: Union[int, Sequence[int]], labels: List[str]) -> pd.DataFrame:
        """
        Retrieve member properties

        :param index: the member index
        :param labels: the member property labels
        :return: datafrome with the selected properties
        """
        return self.member_properties.loc[index, labels]

    def get_member_vectors(self) -> npt.NDArray[np.float_]:
        """
        :return: a 3 x m array with the ``Structure``'s members
        """
        return self.nodes[:, self.members[1, :]] - self.nodes[:, self.members[0, :]]

    def get_member_length(self) -> npt.NDArray[np.float_]:
        """
        :return: an array with the ``Structure``'s member lengths
        """
        return np.linalg.norm(self.nodes[:, self.members[1, :]] - self.nodes[:, self.members[0, :]], axis=0)

    def get_center_of_mass(self) -> npt.NDArray[np.float_]:
        """
        :return: the ``Structure``'s center of mass
        """
        # Computes the center of mass
        mass = self.member_properties['mass'].values
        return np.sum(mass * (self.nodes[:, self.members[0, :]] + self.nodes[:, self.members[1, :]])/2, axis=1) \
            / np.sum(mass)

    def get_centroid(self) -> npt.NDArray[np.float_]:
        """
        :return: the ``Structure``'s geometric center or centroid
        """
        # Computes the centroid (geometric center)
        return np.sum(self.nodes, axis=1) / self.get_number_of_nodes()

    def update_member_properties(self, property_name: Optional[Union[str, Iterable[str]]] = None) -> None:
        """
        Update ``Structure``'s member properties

        If property_name is

        - 'stiffness': calculate ``stiffness`` and ``rest_length`` based on ``modulus``, ``radius`` and ``lambda_``
        - 'mass': calculate ``mass`` and ``volume`` based on ``radius`` and ``density``
        - 'force': calculate ``force`` based on ``lambda_``

        :param property_name: the property name to update; if ``None``, update all properties
        """

        if isinstance(property_name, str):

            # update stiffness and rest length
            if property_name == 'stiffness':
                member_length = self.get_member_length()
                modulus_times_area = self.member_properties['modulus'].values * np.pi * \
                                     (self.member_properties['radius'].values ** 2)
                stiffness = modulus_times_area / member_length
                if np.any(stiffness < 0):
                    raise Exception(f'Structure::update_member_property: negative stiffness computed')
                rest_length = member_length * (1 - self.member_properties['lambda_'].values / stiffness)
                if np.any(rest_length < 0):
                    raise Exception(f'Structure::update_member_property: negative rest_length computed')
                self.member_properties['stiffness'] = stiffness
                self.member_properties['rest_length'] = rest_length

            # update mass and volume
            elif property_name == 'mass':
                member_length = self.get_member_length()
                volume = np.pi * self.member_properties['radius'] ** 2 * member_length
                if np.any(volume < 0):
                    raise Exception(f'Structure::update_member_property: negative volume computed')
                mass = volume * self.member_properties['density']
                if np.any(mass < 0):
                    raise Exception(f'Structure::update_member_property: negative mass computed')
                self.member_properties['volume'] = volume
                self.member_properties['mass'] = mass

            # update force
            elif property_name == 'force':
                self.member_properties['force'] = self.get_member_length() * self.member_properties['lambda_'].values

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
        Returns the set of nodes that lie within ``radius`` distance to other nodes in ``Structure`` and the
        corresponding map

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
        :param verbose: if ``True`` issues a warning with number of nodes removed after the merge
        """
        # get close nodes
        close_nodes_set, close_nodes_map = self.get_close_nodes(radius)
        if close_nodes_set:
            # list of nodes to be removed
            nodes_to_be_removed = list(close_nodes_set)
            # apply merge_map to members
            self.members = close_nodes_map[self.members]
            # remove nodes, to make sure the member node numbering is correct
            self.remove_nodes(nodes_to_be_removed, verify_if_unused=False, verbose=verbose)

    def merge(self, s: 'Structure', inplace=False) -> Optional['Structure']:
        """
        Return a new ``Structure`` in which the current `Structure`` and ``s`` are merged

        :param s: the ``Structure`` to merge
        :param inplace: if ``True`` merge is done in place without creating a copy
        :return: the merged ``Structure``
        """

        if inplace:
            target = self
        else:
            target = self.copy()

        # offset members in s by number_of_members
        number_of_members = self.get_number_of_members()
        number_of_nodes = self.get_number_of_nodes()

        # merge nodes
        target.nodes = np.hstack((self.nodes, s.nodes))

        # merge node tags
        for k, v in s.node_tags.items():
            # append or add
            target.node_tags[k] = np.hstack((self.node_tags[k], number_of_nodes + v)) \
                if k in self.node_tags \
                else number_of_nodes + v

        # merge node properties
        target.node_properties = pd.concat((self.node_properties, s.node_properties), ignore_index=True)

        # merge members, offset by number of nodes
        target.members = np.hstack((self.members, number_of_nodes + s.members))

        # merge member tags
        for k, v in s.member_tags.items():
            # append or add
            target.member_tags[k] = np.hstack((self.member_tags[k], number_of_members + v)) \
                if k in self.member_tags \
                else number_of_members + v

        # merge member properties
        target.member_properties = pd.concat((self.member_properties, s.member_properties), ignore_index=True)

        return target

    def equilibrium(self, force: Optional[npt.ArrayLike] = None, lambda_bar: float = 1,
                    equalities: Optional[list[npt.ArrayLike]] = None,
                    epsilon: float = 1e-7) -> None:
        """
        Solves for the set of internal forces that ensures the equilibrium of the current ``Structure`` in response
        to the vector of external forces ``forces``

        Solve the force equilibrium equation

        .. math::
           A \\lambda = f

        in which:

        - :math:`A`: in a matrix representing the element vectors
        - :math:`f`: is the vector of external forces
        - :math:`\\lambda`: is the vector of resulting force coefficients

        If the ith element is a string then

        .. math::
            x_i \\geq 0

        If no external force is given then the sum of the bar force coefficients equals ``lambda_bar``

        All elements in the equalities are set equal. For example, if::

            equalities = [[0, 1, 3], [2, 4]]

        then the equilibrium is shought satisfying the constraints

        .. math::
            x_0 = x_1 = x_3, \qquad x_2 = x_4

        :param force: a 3 x n array of external forces; or zero if `None`
        :param lambda_bar: the normalizing factor
        :param equalities: a list of lists of member indices which are constrained to have the same force coefficient
        :param epsilon: numerical accuracy
        """

        number_of_nodes = self.get_number_of_nodes()
        number_of_strings = len(self.member_tags.get('string', []))
        number_of_bars = len(self.member_tags.get('bar', []))
        number_of_members = number_of_strings + number_of_bars

        assert number_of_members == number_of_bars + number_of_strings, \
            'number of members is not equal to the sum of number of bars and strings'

        # member vectors
        member_vectors = self.get_member_vectors()

        # coefficient matrix
        Aeq = np.zeros((3 * number_of_nodes, number_of_members))

        # string coefficient
        if number_of_strings:
            for i in self.member_tags['string']:

                ii = 3 * int(self.members[0, i])
                Aeq[ii:ii+3, i] = member_vectors[:, i]

                jj = 3 * int(self.members[1, i])
                Aeq[jj:jj+3, i] = -member_vectors[:, i]

        # bar coefficient
        if number_of_bars:
            for i in self.member_tags['bar']:

                ii = 3 * int(self.members[0, i])
                Aeq[ii:ii+3, i] = -member_vectors[:, i]

                jj = 3 * int(self.members[1, i])
                Aeq[jj:jj+3, i] = member_vectors[:, i]

        A = Aeq
        m = 3 * number_of_nodes

        # external forces
        if force is None:
            A = np.vstack((A, np.ones((1, number_of_members))))
            blo = bup = np.hstack((np.zeros((3 * number_of_nodes,)), 1))
        else:
            beq = np.array(force)
            assert np.all(beq.shape == [3, number_of_nodes]), 'force must be a 3 x n matrix'
            blo = bup = beq.flatten(order='F')

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
        if number_of_strings:
            xlo = np.full((number_of_members, ), 0)
            xlo[self.member_tags['string']] = 0
        else:
            xlo = None

        # cost function
        c = np.ones((number_of_members,))

        # solve lp
        cost, gamma, status = optim.lp(number_of_members, m, c, A, blo, bup, xlo, xup)

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

    def stiffness(self, epsilon: float = 1e-6, storage: str = 'sparse'):
        """
        Computes

        - `v`: potential energy (1 x 1)
        - `F`: force vectors (3 x n)
        - `K`: stiffness matrix (3 n x 3 n)
        - `M`: mass matrix (n x 1)

        for the current ``Structure``

        See also :class:`tensegrity.stiffness.Stiffness`

        :param epsilon: numerical accuracy
        :param storage: if ``sparse`` stores the resulting stiffeness and mass matrices in sparse csr format
        :return: tuple (`v`, `F`, `K`, `M`)
        """

        number_of_nodes = self.get_number_of_nodes()
        number_of_strings = len(self.member_tags.get('string', []))
        number_of_bars = len(self.member_tags.get('bar', []))
        number_of_members = number_of_strings + number_of_bars

        assert number_of_members == number_of_bars + number_of_strings, \
            'number of members is not equal to the sum of number of bars and strings'

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
            K = scipy.sparse.kron(scipy.sparse.coo_matrix((data, row_col),
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
            warnings.warn(f'Structure::stiffness: force balance not satisfied, sum of forces = {sum_of_forces}')

        return v, F, K, M
