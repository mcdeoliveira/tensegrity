from enum import Enum
from typing import Tuple, Union
from typing_extensions import Literal

import numpy as np
import numpy.typing as npt
import scipy


def rotation_2d(phi: float) -> npt.NDArray:
    """
    Return a 2D rotation matrix

    :param phi: the rotation angle in radians
    :return: the rotation matrix
    """
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def rotation_3d(v: npt.NDArray) -> npt.NDArray:
    """
    Return a 3D rotation matrix

    :param v: the vector around which to rotate; its norm is the angle to rotate
    :return: the rotation matrix

    **Notes:**

    1. See :meth:`scipy.spatial.transform.Rotation.from_rotvec` for details
    """
    return scipy.spatial.transform.Rotation.from_rotvec(v).as_matrix()


def orthogonalize(a: Union[npt.NDArray, scipy.sparse.csr_matrix],
                  epsilon: float = 1e-8,
                  mode: Literal['reduced', 'complete'] = 'reduced')\
        -> Union[Tuple[int, npt.NDArray, npt.NDArray], Tuple[int, npt.NDArray]]:
    """
    Ortoghonalize the constraint

    .. math::
        A^T \\, x = 0

    :param a: the coefficient array :math:`A`
    :param epsilon: the accuracy with which to evaluate the rank
    :param mode: 'complete' or 'reduced'
    :return: tuple with rank, the orthogonalized coefficient array, and its null space
             if mode is 'complete'

    **Notes:**

        1. If ``mode = 'reduced'`` the constraint coefficient is normalized at the
           constructor by calculating the *reduced* SVD decomposition

           .. math::
               A = V S U^T, \\quad V^T V = I, \\quad U^T U = I, \\quad
               S \\text{ is diagonal}

           Assuming that :math:`A` is full rank, the equivalent orthogonal constraint

           .. math::
               A^T x = U S V^T x = 0 \\quad \\Leftrightarrow \\quad V^T x = 0

           is obtained in which the coefficient is the orthogonal matrix :math:`V`

        2. If ``mode = 'complete'`` the constraint coefficient is normalized at the
           constructor by calculating the *complete* SVD decomposition

           .. math::
               A = \\tilde{V} \\tilde{S} \\tilde{U}^T, \\quad
               \\tilde{V}^T \\tilde{V} = \\tilde{V} \\tilde{V}^T = I, \\quad
               \\tilde{U}^T \\tilde{U} = \\tilde{U} \\tilde{U}^T = I, \\quad
               \\tilde{S} \\text{ is diagonal}

           Assuming that :math:`A` is full rank, partition

           .. math::
               \\begin{bmatrix} V & T \\end{bmatrix} = \\tilde{V}, \\qquad
               \\begin{bmatrix} U & Q \\end{bmatrix} = \\tilde{U}, \\qquad
               \\tilde{S} = \\begin{bmatrix} S \\\\ 0 \\end{bmatrix}, \\quad
               S \\text{ is diagonal}

           to obtain the equivalent orthogonal constraint and its solution

           .. math::
               A^T x = U S V^T x = 0 \\quad \\Leftrightarrow \\quad V^T x = 0,
               \\qquad x = T \\, y

           The matrix :math:`T` is an orthogonal basis for the constraint null space

        3. The above factorizations are modified to take into account the numerical
           rank of :math:`A` when it is rank-deficient
    """
    assert a.shape[1] < a.shape[0], 'a must be tall'

    # SVD decomposition
    u, s, vh = np.linalg.svd(a.toarray() if scipy.sparse.issparse(a) else a,
                             full_matrices=(mode == 'complete'))

    rank = np.count_nonzero(s > epsilon)
    # return tuple
    if mode == 'complete':
        return rank, u[:, :rank], u[:, rank:]
    else:
        return rank, u[:, :rank]


def is_colinear(a: npt.NDArray, b: npt.NDArray, epsilon: float = 1e-8) -> bool:
    """
    Two vectors are colinear if their orthogonal projection is small

    :param a: first vector
    :param b: second vector
    :param epsilon: accuracy
    :return: ``True`` if colinear
    """
    return np.linalg.norm(a - (np.dot(a, b) / np.dot(b, b)) * b) \
        < epsilon * np.linalg.norm(a)


def norm(a: Union[npt.NDArray, scipy.sparse.csr_matrix]) -> float:
    """
    :param a: the array
    :return: the 2-norm of the array
    """
    return np.linalg.norm(a.data) if scipy.sparse.issparse(a) else np.linalg.norm(a)


class Colors(Enum):
    """
    Default colors
    """

    BLUE = (0, 0.4470, 0.7410)
    """
    """
    ORANGE = (0.8500, 0.3250, 0.0980)
    """
    """
    YELLOW = (0.9290, 0.6940, 0.1250)
    """
    """
    PURPLE = (0.4940, 0.1840, 0.5560)
    """
    """
    GREEN = (0.4660, 0.6740, 0.1880)
    """
    """
    LIGHT_BLUE = (0.3010, 0.7450, 0.9330)
    """
    """
    BROWN = (0.6350, 0.0780, 0.1840)
    """
    """
