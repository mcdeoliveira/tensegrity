from typing import Tuple, Optional

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .plotter import Plotter
from tnsgrt.structure import Structure
from ..utils import Colors


class MatplotlibPlotter(Plotter):
    """
    Matplotlib based structure plotter

    :param plotter: :class:`tnsgrt.plotter.matplotlib.Matplotlib` object
    :param fig: :class:`matplotlib.figure` object
    :param ax: :class:`matplotlib.pyplot.axis` object
    """

    defaults = {
        'plot_nodes': True,
        'plot_members': True,
        'plot_constraints': True,
        'node_marker': 'o',
        'node_markersize': 4,
        'node_linewidth': 2,
        'node_linestyle': 'none',
        'node_facecolor': (0, 0, 0),
        'node_edgecolor': (1, 1, 1),
        'constraint_facecolor': Colors.GREEN.value,
        'constraint_edgecolor': Colors.GREEN.value,
        'constraint_size': 0.1
    }

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
        """
        :return: tuple with matplotlib figure and axis
        """
        return self.fig, self.ax

    def view_init(self, elev=0, azim=0, roll=0) -> None:
        """
        Set view

        :param elev: elevation
        :param azim: azimuth
        :param roll: roll
        """
        self.ax.view_init(elev, azim, roll)

    def plot(self, *s: Structure, **kwargs) -> None:

        # loop if more than one is given
        if len(s) != 1:
            for si in s:
                self.plot(si, **kwargs)
            return
        else:
            s = s[0]

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

        # plot constraints
        if defaults['plot_constraints']:
            facecolor = defaults['constraint_facecolor']
            edgecolor = defaults['constraint_edgecolor']
            size = defaults['constraint_size']
            for j, c in enumerate(s.node_properties['constraint']):
                if c is not None:

                    dofj = c.dof
                    if dofj == 0:
                        # all constrained, plot sphere
                        plot_sphere(self.ax, nodes[:, j],
                                    radius=size/2,
                                    facecolor=facecolor,
                                    edgecolor=edgecolor)

                    elif dofj == 1:
                        # plot line
                        v = size * c.basis[:, 0]
                        plot_line(self.ax,
                                  nodes[:, j] + v,
                                  nodes[:, j] - v,
                                  color=facecolor)

                    elif dofj == 2:
                        # plot plane
                        v = size * c.basis[:, 0]
                        u = size * c.basis[:, 1]
                        plot_parallelogram(self.ax,
                                           nodes[:, j] - v/2 - u/2,
                                           u, v,
                                           facecolor=facecolor,
                                           edgecolor=edgecolor)

        # plot members
        members = s.members
        if defaults['plot_members']:
            for j in range(s.get_number_of_members()):
                if s.member_properties.loc[j, 'visible']:
                    if s.has_member_tag(j, 'string'):
                        # plot strings as lines
                        kwargs = s.get_member_properties(j,
                                                         'facecolor',
                                                         'linewidth').to_dict()
                        kwargs['color'] = kwargs['facecolor']
                        del kwargs['facecolor']
                        plot_line(self.ax,
                                  nodes[:, members[0, j]],
                                  nodes[:, members[1, j]], **kwargs)
                    else:
                        # plot others as solid elements
                        kwargs = s.get_member_properties(j,
                                                         'facecolor', 'edgecolor',
                                                         'volume', 'radius').to_dict()
                        node_i = nodes[:, members[0, j]]
                        node_j = nodes[:, members[1, j]]
                        volume = kwargs.pop('volume')
                        if volume > 0.:
                            kwargs['radius'] = \
                                np.sqrt((volume / np.linalg.norm(node_i - node_j))
                                        / np.pi)

                        plot_solid_cylinder(self.ax, node_i, node_j, **kwargs)

    def plot_arrows(self,
                    origin: npt.NDArray[np.float_], direction: npt.NDArray[np.float_],
                    **kwargs) -> None:
        plot_arrows(self.ax, origin, direction, **kwargs)


def plot_arrows(ax: plt.Axes,
                origin: npt.NDArray[np.float_], direction: npt.NDArray[np.float_],
                arrow_length_ratio: float = 0.2,
                radius: float = 0.01,
                plot_3d_arrows: bool = True,
                **kwargs) -> None:
    if plot_3d_arrows:
        # plot arrows
        for org, dirc in zip(list(map(np.ravel,
                                      np.split(origin, origin.shape[1],
                                               axis=1))),
                             list(map(np.ravel,
                                      np.split(direction, direction.shape[1],
                                               axis=1)))):
            length = np.linalg.norm(dirc)
            if length > 0:
                # plot line
                plot_solid_cylinder(ax, org, org + dirc, radius=radius, **kwargs)
                # plot cone
                plot_truncated_cylinder(ax,
                                        org + (1-arrow_length_ratio)*dirc,
                                        org + dirc,
                                        radius + arrow_length_ratio * length/2,
                                        radius, **kwargs)
    else:
        ax.quiver(*np.split(origin, 3), *np.split(direction, 3),
                  arrow_length_ratio=arrow_length_ratio, **kwargs)


def plot_line(ax: plt.Axes,
              begin: npt.NDArray[np.float_],
              end: npt.NDArray[np.float_],
              **kwargs) -> None:
    """
    Plot line connecting ``x`` to ``y``

    :param ax: :class:`matplotlib.pyplot.axis` object
    :param begin: beginning of line
    :param end: end of line
    :param \**kwargs: additional keywords arguments passed to
                      ``matplotlib.axis.plot``
    """
    # draw lines
    x, y, z = tuple(map(np.ravel, np.split(np.vstack((begin, end)).transpose(), 3)))
    ax.plot(x, y, z, **kwargs)


def plot_parallelogram(ax: plt.Axes,
                       bottom_left: npt.NDArray[np.float_],
                       u: npt.NDArray[np.float_],
                       v: npt.NDArray[np.float_],
                       fill: bool = True,
                       **kwargs) -> None:
    """
    Plot parallelogram from ``bottom_left`` in the directions ``u`` and ``v``

    :param ax: :class:`matplotlib.pyplot.axis` object
    :param bottom_left: bottom left corner of rectangle
    :param v: the ``u`` direction
    :param u: the ``v`` direction
    :param fill: if ``True`` fill area
    :param \**kwargs: additional keywords arguments passed to
                      ``matplotlib.axis.plot``
    """
    vertices = np.vstack((bottom_left, bottom_left + u,
                          bottom_left + u + v, bottom_left + v))

    if fill:
        # # draw surface
        ax.add_collection3d(Poly3DCollection((vertices,), **kwargs))
    else:
        # draw edges only
        ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], **kwargs)


def plot_solid_cylinder(ax: plt.Axes,
                        node_i: npt.NDArray[np.float_],
                        node_j: npt.NDArray[np.float_],
                        radius: float = 0.01,
                        n: int = 12, **kwargs) -> None:
    """
    Plot solid cylinder connecting ``nodes[i]`` to ``nodes[j]``

    :param ax: :class:`matplotlib.pyplot.axis` object
    :param node_i: center of base node
    :param node_j: center of top node
    :param radius: the cylinder radius
    :param n: the number of sides of the cylinder
    :param \**kwargs: additional keywords arguments passed to
                      ``matplotlib.axis.plot``
    """

    # cylinder nodes
    x, y, z = Plotter.cylinder(node_i, node_j, radius, n)

    ax.plot_surface(x, y, z, **kwargs)


def plot_sphere(ax: plt.Axes,
                center: npt.NDArray[np.float_],
                radius: float = 0.01, n: int = 12, **kwargs) -> None:

    # sphere nodes
    x, y, z = Plotter.unit_sphere(n, radius)

    ax.plot_surface(x + center[0], y + center[1], z + center[2], **kwargs)


def plot_truncated_cylinder(ax: plt.Axes,
                            base_center: npt.NDArray[np.float_],
                            top_center: npt.NDArray[np.float_],
                            base_radius: float = 0.01,
                            top_radius: float = 0.01,
                            n: int = 12,
                            **kwargs) -> None:

    # cone nodes
    x, y, z = Plotter.truncated_cylinder(base_center, top_center,
                                         base_radius, top_radius, n)

    ax.plot_surface(x, y, z, **kwargs)
