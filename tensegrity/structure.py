import operator
from dataclasses import dataclass, field
from functools import reduce
from typing import Optional, Dict, get_type_hints, Union, Tuple, Any, List, Hashable, Literal, Sequence
from collections import ChainMap

import numpy as np
import numpy.typing as npt
import pandas as pd


class Structure:

    @dataclass
    class MemberProperty:
        # index
        member_id: int
        # columns
        lmbda: float = 0.
        force: float = 0.
        volume: float = 0.
        mass: float = 1.
        restLength: float = 0.
        # ASTM A36 steel
        yld: float = 250e6
        density: float = 7.85e3
        modulus: float = 200e9
        visible: bool = True
        facecolor: object = (1, 0, 0)
        edgecolor: object = (1, 0, 0)
        linewidth: int = 2
        linestyle: str = '-'

        @staticmethod
        def to_dataframe(data: Union[list, tuple] = tuple()) -> pd.DataFrame:
            # setup member property as pandas dataframe
            hints = get_type_hints(Structure.MemberProperty)
            df = pd.DataFrame(data=data, columns=list(hints.keys())).astype(dtype=hints)
            df.set_index('member_id', inplace=True)
            return df

    member_defaults = {
        'bar': {
            'facecolor': (1, 0, 0),
            'edgecolor': (1, 0, 0)
        },
        'string': {
            'facecolor': (1, 1, 1),
            'edgecolor': (0, 0, 0)
        }
    }

    def __init__(self,
                 nodes: npt.ArrayLike = np.zeros((3, 0), np.float_),
                 members: npt.ArrayLike = np.zeros((2, 0), np.int64),
                 number_of_strings: int = 0,
                 member_tags: Optional[Dict[str, npt.NDArray[np.uint64]]] = None,
                 label: str = None):

        # label, nodes and members
        self.label: Optional[str] = label
        self.nodes: npt.NDArray[np.float_] = np.zeros((3, 0), np.float_)
        self.members: npt.NDArray[np.uint64] = np.zeros((2, 0), np.uint64)
        self.member_tags: Dict[str, npt.NDArray[np.uint64]] = {
            'bar': np.zeros((0,), np.uint64),
            'string': np.zeros((0,), np.uint64)
        }
        self.member_properties: pd.DataFrame = Structure.MemberProperty.to_dataframe()

        # add nodes
        self.add_nodes(nodes)

        # add members
        self.add_members(members, number_of_strings, member_tags)

    def set_nodes(self, nodes: npt.ArrayLike):

        # convert to array
        nodes = np.array(nodes, np.float_)

        # test dimensions
        assert np.equal(nodes.shape, self.nodes.shape), 'nodes shape must match current shape'

        # set nodes
        self.nodes: npt.NDArray[np.float_] = nodes

    def add_nodes(self, nodes: npt.ArrayLike):

        # convert to array
        nodes = np.array(nodes, np.float_)

        # test dimensions
        assert nodes.shape[0] == 3, 'nodes must be a 3 x n array'

        # add nodes
        self.nodes: npt.NDArray[np.float_] = np.hstack((self.nodes, nodes))

    def get_number_of_nodes(self) -> int:
        return self.nodes.shape[1]

    def add_members(self, members: npt.ArrayLike,
                    number_of_strings: Optional[int] = None,
                    member_tags: Optional[Dict[str, npt.NDArray[np.uint64]]] = None):

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
        new_member_properties = [Structure.MemberProperty(i + number_of_members,
                                                          **ChainMap(*[Structure.member_defaults[tag]
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

    def get_number_of_members(self) -> int:
        return self.members.shape[1]

    def get_member_tags(self, i: int) -> List[str]:
        return [k for k, v in self.member_tags.items() if i in v]

    def has_member_tag(self, i: int, tag: str) -> bool:
        return tag in self.member_tags and i in self.member_tags[tag]

    def get_members_by_tags(self, tags: Sequence[str]) -> npt.NDArray[np.uint64]:
        if len(tags) == 0:
            return np.zeros((0,))
        elif len(tags) == 1:
            return self.member_tags.get(tags[0], np.zeros((0,)))
        else:
            return reduce(lambda a1, a2: np.intersect1d(a1, a2, assume_unique=True),
                          [v for k, v in self.member_tags.items() if k in tags])

    def get_member_properties(self, index: Union[int, Sequence[int]], labels: List[str]) -> pd.DataFrame:
        return self.member_properties.loc[index, labels]

    def merge(self, s: 'Structure'):
        # merge Structure s into current structure

        # offset members in s by number_of_members
        number_of_members = self.get_number_of_members()
        number_of_nodes = self.get_number_of_nodes()

        # merge nodes
        self.nodes = np.hstack((self.nodes, s.nodes))

        # merge members, offset by number of nodes
        self.members = np.hstack((self.members, number_of_nodes + s.members))

        # merge member tags
        for k, v in s.member_tags.items():
            if k in self.member_tags:
                # append
                self.member_tags[k] = np.hstack((self.member_tags[k], number_of_members + v))
            else:
                # add
                self.member_tags[k] = number_of_members + v

        # merge member properties
        self.member_properties = pd.concat((self.member_properties, s.member_properties), ignore_index=True)
