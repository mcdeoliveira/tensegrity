import warnings

import numpy as np

from tensegrity.structure import Structure


class Michell(Structure):

    def __init__(self, n: int = 6, beta: float = np.pi/4, q: int = 4,
                 label: str = None, **kwargs):

        # other options
        options = {
            'michell': True,
            'radial': True,
            'outer': True,
            'inner': True,
            'center': True
        }
        options.update(kwargs)

        # wedge angle
        phi = np.pi / n

        # radius ratio
        a = np.sin(beta) / np.sin(beta + phi)

        # warn if not convergent
        if phi + 2 * beta > np.pi:
            warnings.warn('radius are not convergent')

        # nodes
        nodes = np.zeros((3*n, q))
        for i in range(n):
            for k in range(q):
                nodes[3*i:3*(i+1), k] = [a ** k * np.cos((2*i - k) * phi),
                                         a ** k * np.sin((2*i - k) * phi),
                                         0]

        def node_index(i, j):
            return n * j + i

        members = np.zeros((2, 0))
        # add michell members
        if options['michell']:
            michell_members = np.zeros((2*n, 2*(q-1)))
            for i in range(n):
                for j in range(q-1):
                    michell_members[2*i:2*(i+1), j] = [node_index(i, j), node_index(np.mod(i+1, n), np.mod(j+1, q))]
                    michell_members[2*i:2*(i+1), q-1+j] = [node_index(i, j), node_index(i, np.mod(j+1, q))]

            # reshape members
            members = np.hstack((members, michell_members.reshape((2, n*2*(q-1)), order='F')))

        # add radial
        if options['radial'] and q > 2:
            m = q-2
            radial_members = np.zeros((2*n, m))
            for i in range(n):
                for j in range(m):
                    radial_members[2*i:2*(i+1), j] = [node_index(i, j), node_index(np.mod(i+1, n), np.mod(j+2, q))]

            # reshape members
            members = np.hstack((members, radial_members.reshape((2, n*m), order='F')))

        # add outer
        if options['outer']:
            outer_members = np.zeros((2*n, 1))
            for i in range(n):
                outer_members[2*i:2*(i+1), 0] = [node_index(i, 0), node_index(np.mod(i+1, n), 0)]

            # reshape members
            members = np.hstack((members, outer_members.reshape((2, n), order='F')))

        # add inner
        if options['inner']:
            inner_members = np.zeros((2*n, 1))
            for i in range(n):
                inner_members[2*i:2*(i+1), 0] = [node_index(i, q-1), node_index(np.mod(i+1, n), q-1)]

            # reshape members
            members = np.hstack((members, inner_members.reshape((2, n), order='F')))

        # reshape nodes
        nodes = nodes.reshape((3, n * q), order='F')

        # add center
        if options['center']:
            nodes = np.hstack((nodes, np.zeros((3, 1))))
            center_node_index = nodes.shape[1] - 1
            center_members = np.zeros((2*n, 1))
            for i in range(n):
                center_members[2*i:2*(i+1), 0] = [node_index(i, q-1), center_node_index]

            # reshape members
            members = np.hstack((members, center_members.reshape((2, n), order='F')))

        # print(nodes)
        # print(members)

        # remove used options
        kwargs = {k: v for k, v in options.items() if k not in ['michell', 'radial', 'outer', 'inner', 'center']}

        # call super
        super().__init__(nodes, members, label=label, **kwargs)