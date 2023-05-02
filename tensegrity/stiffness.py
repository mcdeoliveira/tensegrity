from typing import Optional
import numpy as np
import numpy.typing as npt
import scipy


class Stiffness:
    # Represents a constrained mechanical model represented by
    #
    #     K: the stiffness matrix
    #     T: a matrix representing node displacements
    #     M: the mass matrix (optional)
    #
    # such that
    #                                        ..    ..
    #     V = y^T fy = (1/2) y^T K y + (1/2) y^T M y,
    #     x = T y
    #
    # Note that the stiffness and mass matrices are stored in the reduced coordinates y but the interface is in terms
    # of the global coordinates x. In global coordinates
    #
    #     fy = T^T fx,    Kx = T^T K T,    Mx = T^T M T
    #
    # The matrix T is often given implicitly as the constraint
    #
    #     R x = 0

    def __init__(self, K: npt.NDArray,
                 M: Optional[npt.NDArray] = None,
                 R: Optional[npt.NDArray] = None, T: Optional[npt.NDArray] = None):

        self.K = K
        self.M = M
        self.T = T
        self.R = R

        assert K.ndim == 2 and K.shape[0] == K.shape[1], 'K must be a square matrix'

        if M is not None:
            assert M.ndim == 2 and M.shape[0] == M.shape[1] and M.shape[0] == K.shape[0], \
                'M must be a square matrix of same size as K'

        assert not(R is not None and T is not None), 'R and T cannot both be given'

        if T is not None:
            assert T.ndim == 2 and T.shape[0] == K.shape[0], 'T must have the same number of rows as K'

        if R is not None:
            # apply constraint
            self.apply_constraint(R)

    def apply_constraint(self, R: npt.NDArray, local: bool = True):
        # Apply the constraint
        #
        #     R y = 0,    x = T y
        #
        # to the current stiffness object.
        #
        # All solutions to the constraint are given by
        #
        #     y = V z,    R V = 0,    V^T V > 0
        #
        # The new model after applying the constraint is
        #
        #     V = z^T fz = (1/2) z^T Kz z,   x = Tz z
        #
        # in which
        #
        #     fz = T^T f,    Kz = T^T K T,    Tz = T V
        #
        # If 'local' = False then the constraint is considered on the global coordinate
        #
        #     R x = 0,    x = T y
        #
        # which correspond to the constraint on the local coordinate
        #
        #     R T y = 0

        if not local:
            if self.T is not None:
                assert R.ndim == 2 and R.shape[1] == self.T.shape[0], 'R must have the same number of columns as T'
                R = R @ self.T

        assert R.ndim == 2 and R.shape[1] == self.K.shape[0], 'R must have the same number of columns as K'

        # calculate null space
        V = scipy.linalg.null_space(R)

        if scipy.sparse.issparse(self.K):
            V = scipy.sparse.csr_matrix(V)

        # project stiffness
        self.K = V.transpose() @ self.K @ V

        # symmetrize
        self.K += self.K.transpose()
        self.K /= 2

        if self.M is not None:
            # project mass
            self.M = V.transpose() @ self.M @ V
            self.M += self.M.transpose()
            self.M /= 2

        if self.T is None:
            self.T = V
        else:
            self.T = V @ self.T

    def displacements(self, f: npt.NDArray):
        # Calculate global displacements due to the global force f
        #
        #     x = T K^(-1) T^T f

        return self.T @ np.linalg.solve(self.K, self.T.transpose() @ f) \
            if self.T is not None else np.linalg.solve(self.K, self.T.transpose() @ f)

    def eigs(self, k: int = 12):
        # compute the (k smallest) eigenvalues and associated eigenvectors of the (sparse) stiffness matrix

        if scipy.sparse.issparse(self.K):
            # calculate sparse eigenvalues
            d, V = scipy.sparse.linalg.eigsh(self.K, k=k, which='SM')

        else:
            # calculate dense eigenvalues
            d, V = np.linalg.eigh(self.K)

        # [d, dInd] = np.sort(d)
        # sort eigs
        ind = np.argsort(d)

        # project eigenvectors
        if self.T is not None:
            V = self.T @ V

        return d[ind], V[:, ind]

    def eigenmodes(self, k: int = 12, units='Hz'):
        # compute the natural frequencies[rad/s] and eigenvectors for the stiffness

        if self.M is None:
            raise 'modes cannot be calculated without mass information'

        if scipy.sparse.issparse(self.K):
            # calculate sparse eigenvalues
            d, V = scipy.sparse.linalg.eigsh(self.K, k=k, M=self.M, which='SM')
        else:
            # calculate dense eigenvalues
            d, V = scipy.linalg.eigh(self.K, self.M)

        # [d, dInd] = np.sort(d)
        # sort eigs
        ind = np.argsort(d)

        # project eigenvectors
        if self.T is not None:
            V = self.T @ V

        # units
        convert = 1/(2*np.pi) if units == 'Hz' else 1

        return convert * np.sqrt(d[ind]), V[:, ind]

    @staticmethod
    def rigid_body_constraint(nodes: npt.NDArray) -> npt.NDArray:
        # Construct rigid body constraint matrix R x = 0 in global coordinates

        assert len(nodes.shape) == 2 and nodes.shape[0] == 3, 'nodes must be a 3 x m matrix'

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
                          r3 / np.linalg.norm(r3)))