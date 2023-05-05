from typing import Union, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from tensegrity.structure import Structure


class Plotter:

    def plot(self, s: Union[Structure, Sequence[Structure]], **kwargs):
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
        x, y, z = Plotter.unit_cylinder(n, radius, length)

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
