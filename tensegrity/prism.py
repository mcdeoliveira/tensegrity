import numpy as np

from .structure import Structure
from .utils import rotation_2d


class Prism(Structure):

    def __init__(self, p: int = 3,
                 top_radius: float = 1, bottom_radius: float = 1, height: float = 1,
                 label: str = None, calculate_equilibrium=True, equilibrium_method='analytic',
                 **_options):

        # proper size
        assert p >= 3, 'p must be greater or equal to 3'

        # twist angle
        if 'alpha' in _options:
            alpha = _options.pop('alpha')
        else:
            alpha = np.pi/2 - np.pi/p

        # other options
        options = {
            'bottom': True,
            'top': True,
            'vertical': True,
            'diagonal': False,
            'bar': True
        }
        options.update(_options)

        # valid equilibrium?
        if calculate_equilibrium:
            assert 'bottom' in options, 'equilibrium is not possible without bottom strings'
            assert 'top' in options, 'equilibrium is not possible without top strings'
            assert 'bar' in options, 'equilibrium is not possible without bars'
            if not options['diagonal']:
                assert np.abs(np.pi/2 - np.pi/p - alpha) < 1e-8, 'equilibrium is not possible with given twist angle'
            if not options['vertical']:
                assert np.abs(np.pi/2 - alpha) < 1e-8, 'equilibrium is not possible with given twist angle'
            if options['vertical'] and options['diagonal']:
                assert np.pi/2 - np.pi/p <= alpha <= np.pi/2, 'equilibrium is not possible with given twist angle'

        # base angle
        beta = 2 * np.pi/p

        # rotation matrices
        rotation_beta = rotation_2d(beta)
        rotation_alpha = rotation_2d(alpha)

        # nodes
        nodes = np.zeros((3, 2 * p))
        # bottom nodes
        nodes[2, :p] = -height / 2
        nodes[:2, 0] = np.array([bottom_radius, 0])
        for i in range(1, p):
            nodes[:2, i] = rotation_beta @ nodes[:2, i-1]
        # top nodes
        nodes[2, p:2 * p] = height / 2
        nodes[:2, p] = np.array([top_radius, 0])
        nodes[:2, p] = rotation_alpha @ nodes[:2, p]
        for i in range(1, p):
            nodes[:2, p + i] = rotation_beta @ nodes[:2, p + i - 1]

        # strings
        members = np.zeros((2,0))
        number_of_strings = 0
        string_tags = {
        }
        # bottom strings
        bottom_strings = np.vstack((np.arange(p), np.mod(1 + np.arange(p), p)))
        if options['bottom']:
            members = np.hstack((members, bottom_strings))
            string_tags['bottom'] = number_of_strings + np.arange(p)
            number_of_strings += p
        # top strings
        top_strings = np.vstack((p + np.arange(p), p + np.mod(1 + np.arange(p), p)))
        if options['top']:
            members = np.hstack((members, top_strings))
            string_tags['top'] = number_of_strings + np.arange(p)
            number_of_strings += p
        # vertical strings
        vertical_strings = np.vstack((np.arange(p), p + np.arange(p)))
        if options['vertical']:
            members = np.hstack((members, vertical_strings))
            string_tags['vertical'] = number_of_strings + np.arange(p)
            number_of_strings += p
        # diagonal strings
        diagonal_strings = np.vstack((np.arange(p), p + np.mod(np.arange(p) - 1, p)))
        if options['diagonal']:
            members = np.hstack((members, diagonal_strings))
            string_tags['diagonal'] = number_of_strings + np.arange(p)
            number_of_strings += p

        # bars
        bars = np.vstack((np.arange(p), p + np.mod(1 + np.arange(p), p)))
        if options['bar']:
            members = np.hstack((members, bars))

        # call super
        super().__init__(nodes=nodes, members=members, number_of_strings=number_of_strings,
                         member_tags=string_tags, label=label)

        if calculate_equilibrium:
            if equilibrium_method == 'analytic':
                # top/bottom ratio
                rho = top_radius / bottom_radius
                # force coefficients
                lmbda = np.array([
                    rho * np.cos(np.pi / p) / np.cos(alpha - np.pi / p),                # bottom
                    (1 / rho) * np.cos(np.pi / p) / np.cos(alpha - np.pi / p),          # top
                    2 * np.cos(alpha) * np.cos(np.pi / p) / np.cos(alpha - np.pi / p),  # vertical
                    -np.cos(alpha + np.pi / p) / np.cos(alpha - np.pi / p),             # diagonal
                    -1                                                                  # bars
                ]).reshape((1, 5))
                # select appropriate string sets
                index = [True, True, options['vertical'], options['diagonal'], True]
                # construct coefficients
                self.member_properties['lmbda'] = (np.ones((p, 1)) @ lmbda[:, index]).flatten(order='F')
            else:
                # solve for equilibrium
                self.equilibrium(equalities=[np.arange(number_of_strings, members.shape[1])])

            # update mass, volume, stiffness and rest length
            self.update_member_properties()
