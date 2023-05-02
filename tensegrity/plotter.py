from typing import Tuple, Optional, Union, Sequence

from .structure import Structure
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


class Plotter:

    def plot(self, s: Structure, **kwargs):
        pass

    @staticmethod
    def unit_cylinder(n: int = 10, radius: float = 1, height: float = 1):
        z = np.linspace(0, height, 2)
        theta = np.linspace(0, 2 * np.pi, n)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)
        return x_grid, y_grid, z_grid

    @staticmethod
    def cylinder(node1: npt.NDArray, node2: npt.NDArray,
                 volume: float = 0., radius: float = 0.01, n: int = 12) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:

        # rod vector length
        rod = node2 - node1
        length = np.linalg.norm(rod)
        if volume > 0.:
            radius = np.sqrt((volume / length) / np.pi)

        # draw unit cylinder
        x, y, z = MatplotlibPlotter.unit_cylinder(n, radius, length)

        # rotation vector
        rotation_vector = [-rod[1], rod[0], 0]

        # rotation angle
        angle = np.arccos(rod[2] / length)
        if angle > 1e-4:

            norm = np.linalg.norm(rotation_vector)
            if norm < 1e-6:
                rotation_vector = [1, 0, 0]
            else:
                rotation_vector /= norm

            rotation_matrix = Rotation.from_rotvec(angle * rotation_vector)

            # rotate
            for i in range(x.shape[0]):
                xyz = np.vstack((x[i, :], y[i, :], z[i, :]))
                xyz = rotation_matrix.apply(xyz.transpose()).transpose()
                x[i, :] = xyz[0, :]
                y[i, :] = xyz[1, :]
                z[i, :] = xyz[2, :]

        # translate
        x += node1[0]
        y += node1[1]
        z += node1[2]

        return x, y, z


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

    def plot(self, s: Union[Structure, Sequence[Structure]], **kwargs):

        # loop if a sequence is given
        if not isinstance(s, Structure):
            for si in s:
                self.plot(si)
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
