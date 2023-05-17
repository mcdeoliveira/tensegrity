from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from tnsgrt.structure import Structure


class Plotter:
    """
    Base class for structure plotters
    """

    def plot(self, *s: Structure, **kwargs) -> None:
        """
        Plot structure

        :param \\*s: the structure or sequence of structures
        :param \\**kwargs: additional keyword arguments
        """
        pass

    @staticmethod
    def unit_cylinder(n: int = 10, radius: float = 1, height: float = 1) \
            -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Return xyz grid points for solid cylinder centered at the origin

        :param n: number of sides
        :param radius: radius of cylinder
        :param height: height of cylinder
        :return: tuple with x, y, and z points
        """
        z = np.linspace(0, height, 2)
        theta = np.linspace(0, 2 * np.pi, n)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)
        return x_grid, y_grid, z_grid

    @staticmethod
    def cylinder(node1: npt.NDArray, node2: npt.NDArray,
                 volume: float = 0., radius: float = 0.01, n: int = 12) \
            -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Return xyz grid points for cylinder with center aligned with the vector defined
        by ``node1`` and ``node2``

        :param node1: center of the base of the cylinder
        :param node2: center of the top of the cylinder
        :param volume: volume of the cylinder, used to set the radius if positive
        :param radius: radius of the cylinder, ignore if volume is positive
        :param n: number of sides
        :return: tuple with x, y, and z points
        """
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
                rotation_vector = np.array([1, 0, 0])
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
