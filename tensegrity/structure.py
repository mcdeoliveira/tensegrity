from dataclasses import dataclass, field
from functools import reduce
from typing import Optional, Dict, get_type_hints, Union, Tuple

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
            df.set_index('member_id')
            return df

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

    def add_members(self, new_members: npt.ArrayLike,
                    number_of_new_strings: Optional[int] = None,
                    new_member_tags: Optional[Dict[str, npt.NDArray[np.uint64]]] = None):

        # convert to array
        new_members = np.array(new_members, np.uint64)

        # test dimensions
        assert new_members.shape[0] == 2, 'members must be a 2 x m array'

        # member tags
        number_of_new_members = new_members.shape[1]
        if new_member_tags is not None:
            # make sure member tags are unique
            for k, v in new_member_tags.items():
                new_member_tags[k] = np.unique(v)
                assert np.amin(v) >= 0, \
                    'member tag index must be greater or equal than zero'
                assert np.amax(v) < number_of_new_members, \
                    'member index must be less than number of members'
            # make sure bars and strings are mutually exclusive
            assert 'bar' not in new_member_tags or 'string' not in new_member_tags or \
                np.intersect1d(new_member_tags['bar'], new_member_tags['string'], assume_unique=True).size == 0, \
                'bar and string tags must be mutually exclusive'
        elif number_of_new_strings is not None:
            # number of strings given
            assert number_of_new_strings <= number_of_new_members, \
                'number of added strings must be less than number of added members'
            number_of_new_bars = number_of_new_members - number_of_new_strings
            new_member_tags = {
                'string': np.arange(0, number_of_new_strings, dtype=np.uint64),
                'bar': number_of_new_strings + np.arange(0, number_of_new_bars, dtype=np.uint64)
            }
        else:
            raise Exception('Either type or number of strings must be provided')

        # new member properties
        number_of_members = self.get_number_of_members()
        new_member_properties = [Structure.MemberProperty(i + number_of_members) for i in range(new_members.shape[1])]

        # make sure member index is valid
        number_of_nodes = self.get_number_of_nodes()
        assert number_of_new_members == 0 or np.amin(new_members) >= 0, \
            'member index must be greater or equal than zero'
        assert number_of_new_members == 0 or np.amax(new_members) < number_of_nodes, \
            'member index must be less than number of nodes'

        # add new members
        self.members = np.hstack((self.members, new_members))
        # add member tags
        for k, v in new_member_tags.items():
            self.member_tags[k] = np.hstack((self.member_tags[k], number_of_members + v)) \
                if k in self.member_tags else number_of_members + v

        # add default member properties
        self.member_properties = pd.concat((self.member_properties,
                                            Structure.MemberProperty.to_dataframe(new_member_properties)))

    def get_number_of_members(self) -> int:
        return self.members.shape[1]

    def get_member_tags(self, i: int):
        return [k for k, v in self.member_tags.items() if i in v]

    def has_member_tag(self, i: int, tag: str):
        return tag in self.member_tags and i in self.member_tags[tag]

    def get_element_by_tags(self, tags: Tuple[str]):
        if len(tags) == 0:
            return np.zeros((0,))
        elif len(tags) == 1:
            return self.member_tags.get(tags[0], np.zeros((0,)))
        else:
            return reduce(lambda a1, a2: np.intersect1d(a1, a2, assume_unique=True),
                          [v for k, v in self.member_tags.items() if k in tags])

