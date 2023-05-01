import warnings
from typing import Optional
import numpy as np
import numpy.typing as npt
import scipy


def rigid_body_constraint(nodes: npt.NDArray):
    # Construct rigid body constraint matrix R x = 0

    assert len(nodes.shape) == 2 and nodes.shape[0] == 3

    number_of_nodes = nodes.shape[1]

    # rigid translation
    t1 = np.zeros((3, number_of_nodes), order='F')
    t1[0, :] = 1 / np.sqrt(number_of_nodes)

    t2 = np.zeros((3, number_of_nodes), order='F')
    t2[1, :] = 1 / np.sqrt(number_of_nodes)

    t3 = np.zeros((3, number_of_nodes), order='F')
    t3[2, :] = 1 / np.sqrt(number_of_nodes)

    translation = np.vstack((t1.ravel(order='F'), t2.ravel(order='F'), t3.ravel(order='F')))

    # rigid rotation
    rotation1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    rotation3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

    r1 = rotation1 @ nodes
    r2 = rotation2 @ nodes
    r3 = rotation3 @ nodes

    r1 = r1.ravel(order='F')
    r2 = r2.ravel(order='F')
    r3 = r3.ravel(order='F')

    return np.vstack((translation,
                      r1 / np.linalg.norm(r1),
                      r2 / np.linalg.norm(r2),
                      r3 / np.linalg.norm(r3))).transpose()


def apply_constraint(R: npt.NDArray, K: npt.NDArray, T: Optional[npt.NDArray] = None, M: Optional[npt.NDArray] = None):
    # apply the constraint
    #
    # V = x^T f = (1/2) x^T K x, R^T x = 0
    #
    # R^T Rp = 0 => x = Rp y
    #
    # V = y^T (Rp^T f) = (1/2) y^T (Rp^T K Rp) y

    # calculate null space
    Rp = scipy.linalg.null_space(R.transpose())

    # project stiffness
    K = Rp.transpose() @ K @ Rp

    # symmetrize
    K += K.transpose()
    K /= 2

    if M is not None:
        # project mass
        M = Rp.transpose() @ M @ Rp
        M += M.transpose()
        M /= 2

    return K, T @ Rp if T is not None else Rp, M


def displacements(f: npt.NDArray, K: npt.NDArray, T: Optional[npt.NDArray] = None):
    # Calculate displacements due to force f
    #
    #     x = K^(-1) f
    #
    # If there are constraints
    #
    #     x = T K^(-1) T^T f

    return T @ np.linalg.solve(K, T.transpose() @ f) if T is not None else np.linalg.solve(K, T.transpose() @ f)


def eigenmodes(K: npt.NDArray, M: npt.NDArray, T: Optional[npt.NDArray] = None):
    # compute the natural frequencies[rad/s] and eigenvectors for the stiffness

    # neig = K.shape[0]

    # opts.disp = 0;
    # opts.issym = 1;

    M = scipy.linalg.sqrtm(M)
    # print(M)
    K = M @ K @ M
    # print(K)

    d, V = np.linalg.eigh(K)   # neig, 'SM', opts
    if np.any(d < 0):
        if np.min(d) < -1e-8* np.max(d):
            raise 'negative eigenvalues'
        warnings.warn('small negative eigenvalues')
        d = np.abs(d)

    # [d, dInd] = np.sort(d)
    ind = np.argsort(d)

    if T is not None:
        V = T @ V

    return np.sqrt(d[ind]), V[:, ind]
