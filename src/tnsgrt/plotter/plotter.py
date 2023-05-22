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
            -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
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
    def unit_truncated_cylinder(n: int = 10,
                                base_radius: float = 1, top_radius: float = 1,
                                height: float = 1) \
            -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Return xyz grid points for solid truncated cylinder centered at the origin

        :param n: number of sides
        :param base_radius: radius of the cylinder base
        :param top_radius: radius of the cylinder top
        :param height: height of cylinder
        :return: tuple with x, y, and z points
        """
        z = np.linspace(0, height, 2)
        theta = np.linspace(0, 2 * np.pi, n)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = (base_radius * (1 - z_grid/height) +
                  top_radius * z_grid/height) * np.cos(theta_grid)
        y_grid = (base_radius * (1 - z_grid/height) +
                  top_radius * z_grid/height) * np.sin(theta_grid)
        return x_grid, y_grid, z_grid

    @staticmethod
    def unit_sphere(n: int = 10, radius: float = 1) \
            -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Return xyz grid points for solid sphere centered at the origin

        :param n: number of sides
        :param radius: radius of sphere
        :return: tuple with x, y, and z points
        """

        phi = np.linspace(0, np.pi, n)
        theta = np.linspace(0, 2 * np.pi, 2*n)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        x_grid = radius * np.cos(theta_grid) * np.sin(phi_grid)
        y_grid = radius * np.sin(theta_grid) * np.sin(phi_grid)
        z_grid = radius * np.cos(phi_grid)
        return x_grid, y_grid, z_grid

    @staticmethod
    def truncated_cylinder(node1: npt.NDArray,
                           node2: npt.NDArray,
                           base_radius: float = 0.01, top_radius: float = 0.01,
                           n: int = 12) \
            -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Return xyz grid points for cylinder with center aligned with the vector defined
        by ``node1`` and ``node2``

        :param node1: center of the base of the cylinder
        :param node2: center of the top of the cylinder
        :param base_radius: radius of the cylinder base
        :param top_radius: radius of the cylinder top
        :param n: number of sides
        :return: tuple with x, y, and z points
        """
        # cylinder vector length
        axis = node2 - node1
        height = np.linalg.norm(axis)

        # draw unit cylinder
        x, y, z = Plotter.unit_truncated_cylinder(n, base_radius, top_radius, height)

        # rotation vector
        rotation_vector = [-axis[1], axis[0], 0]

        # rotation angle
        angle = np.arccos(axis[2] / height)
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

    @staticmethod
    def cylinder(node1: npt.NDArray, node2: npt.NDArray,
                 radius: float = 0.01, n: int = 12) \
            -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Return xyz grid points for cylinder with center aligned with the vector defined
        by ``node1`` and ``node2``

        :param node1: center of the base of the cylinder
        :param node2: center of the top of the cylinder
        :param radius: radius of the cylinder, ignore if volume is positive
        :param n: number of sides
        :return: tuple with x, y, and z points
        """

        # draw unit cylinder
        return Plotter.truncated_cylinder(node1, node2, radius, radius, n)

    @staticmethod
    def cone(node1: npt.NDArray, node2: npt.NDArray,
             base_radius: float = 0.01, n: int = 12) \
            -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Return xyz grid points for cylinder with center aligned with the vector defined
        by ``node1`` and ``node2``

        :param node1: center of the base of the cylinder
        :param node2: center of the top of the cylinder
        :param base_radius: radius of the cylinder, ignore if volume is positive
        :param n: number of sides
        :return: tuple with x, y, and z points
        """
        return Plotter.truncated_cylinder(node1, node2, base_radius, 0, n)
