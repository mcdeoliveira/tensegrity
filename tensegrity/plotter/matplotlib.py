from typing import Tuple, Optional, Union, Sequence

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from .plotter import Plotter
from tensegrity.structure import Structure


class MatplotlibPlotter(Plotter):
    defaults = {
        'plot_nodes': True,
        'plot_members': True,
        'node_marker': 'o',
        'node_markersize': 4,
        'node_linewidth': 2,
        'node_linestyle': 'none',
        'node_facecolor': (0, 0, 0),
        'node_edgecolor': (1, 1, 1)
    }

    @staticmethod
    def plot_line(ax: plt.Axes, nodes: npt.NDArray[np.float_], i: int, j: int, **kwargs):
        # draw lines
        x = np.hstack((nodes[0, i], nodes[0, j]))
        y = np.hstack((nodes[1, i], nodes[1, j]))
        z = np.hstack((nodes[2, i], nodes[2, j]))
        ax.plot(x, y, z, **kwargs)

    @staticmethod
    def plot_solid_cylinder(ax: plt.Axes, nodes: npt.NDArray[np.float_], i: int, j: int,
                            volume: float = 0., radius: float = 0.01, n: int = 12,
                            **kwargs):

        # cylinder nodes
        x, y, z = Plotter.cylinder(nodes[:, j], nodes[:, i], volume, radius, n)

        ax.plot_surface(x, y, z, **kwargs)

    def __init__(self, plotter: Optional['MatplotlibPlotter'] = None,
                 fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None):
        # call super
        super().__init__()

        # initialize plotter
        if plotter is not None:
            if ax is not None or fig is not None:
                raise 'fig and ax handles can not be given when plotter is given'
            self.fig, self.ax = plotter.get_handles()

        elif fig is None:
            if ax is not None:
                raise 'fig handle must be given when ax is given'
            # initialize figure and axis
            self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})

        elif ax is None:
            # initialize axis
            self.ax = fig.subplots(subplot_kw={"projection": "3d"})

        else:
            # assign fig and axis
            self.fig, self.ax = fig, ax

    def get_handles(self) -> Tuple[plt.Figure, plt.Axes]:
        return self.fig, self.ax

    def view_init(self, elev=0, azim=0):
        self.ax.view_init(elev, azim)

    def plot(self, s: Union[Structure, Sequence[Structure]], **kwargs):

        # loop if a sequence is given
        if not isinstance(s, Structure):
            for si in s:
                self.plot(si, **kwargs)
            return

        # process options
        defaults = MatplotlibPlotter.defaults.copy()
        defaults.update(kwargs)

        # plot nodes
        nodes = s.nodes
        if defaults['plot_nodes']:
            self.ax.plot(nodes[0, :], nodes[1, :], nodes[2, :],
                    defaults['node_marker'],
                    markersize=defaults['node_markersize'],
                    linewidth=defaults['node_linewidth'],
                    linestyle=defaults['node_linestyle'],
                    markerfacecolor=defaults['node_facecolor'],
                    markeredgecolor=defaults['node_edgecolor'])

        # plot members
        members = s.members
        if defaults['plot_members']:
            for j in range(s.get_number_of_members()):
                if s.member_properties.loc[j, 'visible']:
                    if s.has_member_tag(j, 'string'):
                        # plot strings as lines
                        kwargs = s.get_member_properties(j, ['facecolor', 'linewidth']).to_dict()
                        kwargs['color'] = kwargs['facecolor']
                        del kwargs['facecolor']
                        MatplotlibPlotter.plot_line(self.ax, nodes, members[0, j], members[1, j], **kwargs)
                    else:
                        # plot others as solid elements
                        kwargs = s.get_member_properties(j, ['facecolor', 'edgecolor', 'volume']).to_dict()
                        MatplotlibPlotter.plot_solid_cylinder(self.ax, nodes, members[0, j], members[1, j], **kwargs)
