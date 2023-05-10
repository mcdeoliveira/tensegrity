import warnings
from typing import Optional, Tuple, Sequence, Union, Literal
import numpy as np
import numpy.typing as npt
import scipy

from tensegrity.utils import orthogonalize


class NodeConstraint:
    """
    Object representing a node constraint

    :param constraint: the constraint coefficient array
    :param epsilon: precision used to assess numerical rank

    **Notes:**

    1. If :math:`x` is the node coordinate and :math:`A` is the constraint coefficient then

       .. math::
           A x = 0

       then the equivalent constraint and its null space

       .. math::
           R \\, x = 0, \\quad x = T \\, y, \\quad R R^T = I, \\quad T^T T = I

       is constructed using :func:`tensegrity.utils.orthogonalize`
    """

    def __init__(self, constraint: Optional[npt.NDArray] = None, epsilon: float = 1e-8):
        if constraint is None:
            self.dof = 0
            self.basis = None
            self.normal = np.eye(3)
        else:
            assert constraint.ndim == 2 and constraint.shape[1] == 3 and constraint.shape[0] < 3, \
                'normal must be a m x 3, m < 3, array'
            # first orthogonalize
            rank, r, t = orthogonalize(constraint.transpose(), mode='complete', epsilon=epsilon)
            self.dof = 3 - rank
            if self.dof == 0:
                self.basis = None
                self.normal = np.eye(3)
            else:
                self.normal = r.transpose()
                self.basis = t

    @staticmethod
    def node_constraint(nodes: npt.NDArray, constraints: Sequence['NodeConstraint'], storage: str = 'sparse') \
            -> Union[Tuple[npt.NDArray, npt.NDArray], Tuple[scipy.sparse.csr_array, scipy.sparse.csr_array]]:
        """
        Construct the constraint associated with the given nodes and node constraints

        The resulting tuple is compatible with :meth:`~tensegrity.stiffness.Stiffness.apply_constraint`

        :param nodes: 3 x n array of nodes
        :param constraints: list with n constraints
        :param storage: if ``sparse``, returns sparse arrays
        :return: tuple with the constraint matrix, and its null space
        """
        assert nodes.shape[0] == 3, 'nodes must be a 3 x m array'
        assert nodes.shape[1] == len(constraints), \
            'number of columns of nodes must be equal to to the number of constraints'

        # calculate degrees of freedom
        m = nodes.shape[1]
        noc = sum(3 - c.dof for c in constraints if c is not None)
        dof = 3*m - noc

        if storage == 'sparse':
            row_col_r = np.zeros((2, 3*noc))
            data_r = np.zeros((3*noc))
            row_col_t = np.zeros((2, 3*dof))
            data_t = np.zeros((3*dof))
            jj = kk = 0
            for i, c in enumerate(constraints):
                if c is None:
                    # add identity to basis
                    dofj = 3
                    row_col_t[0, 3*kk:3*kk+3*dofj] = np.kron(np.arange(3*i, 3*(i+1)), np.ones((3,)))
                    row_col_t[1, 3*kk:3*kk+3*dofj] = np.kron(np.arange(kk, kk+dofj), np.ones((3,)))
                    data_t[3*kk:3*kk+3*dofj] = np.eye(dofj).flatten(order='C')
                    kk += dofj
                else:
                    # add to a
                    nocj = 3-c.dof
                    row_col_r[0, 3*jj:3*jj+3*nocj] = np.kron(np.arange(jj, jj+nocj), np.ones((3,)))      # row
                    row_col_r[1, 3*jj:3*jj+3*nocj] = np.kron(np.ones((nocj,)), np.arange(3*i, 3*(i+1)))  # col
                    data_r[3*jj:3*jj+3*nocj] = c.normal.flatten(order='C')
                    jj += nocj
                    if c.dof:
                        # add to basis
                        row_col_t[0, 3*kk:3*kk+3*c.dof] = np.kron(np.arange(3*i, 3*(i+1)), np.ones((c.dof,)))  # row
                        row_col_t[1, 3*kk:3*kk+3*c.dof] = np.kron(np.arange(kk, kk+c.dof), np.ones((3,)))      # col
                        data_t[3*kk:3*kk+3*c.dof] = c.basis.flatten(order='C')
                        kk += c.dof
            R = scipy.sparse.coo_matrix((data_r, row_col_r),
                                        shape=(noc, 3*m)).tocsr()
            T = scipy.sparse.coo_matrix((data_t, row_col_t),
                                        shape=(3*m, dof)).tocsr()
        else:
            R = np.zeros((noc, 3*m))
            T = np.zeros((3*m, dof))
            jj = kk = 0
            for i, c in enumerate(constraints):
                if c is None:
                    # add identity to basis
                    T[3 * i:3 * (i + 1), kk:kk+3] = np.eye(3)
                    kk += 3
                else:
                    # add to a
                    nocj = 3-c.dof
                    R[jj:jj+nocj, 3*i:3*(i+1)] = c.normal
                    jj += nocj
                    if c.dof:
                        # add to basis
                        T[3*i:3*(i+1), kk:kk+c.dof] = c.basis
                        kk += c.dof

        return R, T

    @staticmethod
    def rigid_body_constraint(nodes: npt.NDArray, epsilon: float = 1e-8) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Construct rigid-body constraint associated with the given nodes

        The resulting tuple is compatible with :meth:`~tensegrity.stiffness.Stiffness.apply_constraint`

        :param nodes: 3 x n array of nodes
        :param epsilon: precision used to assess numerical rank
        :return: tuple with the constraint matrix, and its null space
        """

        assert len(nodes.shape) == 2 and nodes.shape[0] == 3, 'nodes must be a 3 x m matrix'

        number_of_nodes = nodes.shape[1]

        # rigid translation
        t1 = np.zeros((3, number_of_nodes), order='F')
        t1[0, :] = 1 / np.sqrt(number_of_nodes)

        t2 = np.zeros((3, number_of_nodes), order='F')
        t2[1, :] = 1 / np.sqrt(number_of_nodes)

        t3 = np.zeros((3, number_of_nodes), order='F')
        t3[2, :] = 1 / np.sqrt(number_of_nodes)

        # rigid rotations
        r1 = (np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]]) @ nodes).ravel(order='F')
        r2 = (np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]) @ nodes).ravel(order='F')
        r3 = (np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) @ nodes).ravel(order='F')

        r = np.vstack((t1.ravel(order='F'), t2.ravel(order='F'), t3.ravel(order='F'),
                       r1 / np.linalg.norm(r1), r2 / np.linalg.norm(r2), r3 / np.linalg.norm(r3)))

        rank, r, t = orthogonalize(r.transpose(), epsilon=epsilon, mode='complete')

        return r.transpose(), t


class Stiffness:
    """
    Model for a constrained mechanical system consisting of

    - :math:`K`: the stiffness matrix
    - :math:`M`: the mass matrix
    - :math:`R`: a matrix representing node displacement constraints
    - :math:`T`: a matrix representing allowed node displacements

    such that

    .. math::
        V = \\frac{1}{2} y^T K y + \\frac{1}{2} \\ddot{y}^T M \\ddot{y}, \\qquad R \\, x = 0, \\qquad x = T y

    Constructor parameters:

    :param K: the stiffness matrix
    :param M: the mass matrix

    **Notes:**

    1. The stiffness and mass matrices are stored in the reduced coordinates :math:`y`

    2. In global coordinates

       .. math::
           V = \\frac{1}{2} x^T K_x x + \\frac{1}{2} \\ddot{x}^T M_x \\ddot{x}

       in which

       .. math::
           K_x = T^T K T, \\qquad M_x = T^T M T

    3. Use :meth:`tensegrity.stiffness.Stiffness.apply_constraint` to apply constraints to the model

    4. Node constraints are enforced to be orthogonal so that

       .. math::
           R \\, x = 0, \\quad x = T y, \\qquad R R^T = I, \\quad T^T T = I, \\quad R \\, T = 0
    """

    def __init__(self,
                 K: Union[npt.NDArray, scipy.sparse.csr_matrix],
                 M: Optional[Union[npt.NDArray, scipy.sparse.csr_matrix]] = None):

        self.K = K
        self.M = M
        self.T: Optional[Union[npt.NDArray, scipy.sparse.csr_matrix]] = None
        self.R: Optional[Union[npt.NDArray, scipy.sparse.csr_matrix]] = None

        assert K.ndim == 2 and K.shape[0] == K.shape[1], 'K must be a square matrix'

        if M is not None:
            assert M.ndim == 2 and M.shape[0] == M.shape[1] and M.shape[0] == K.shape[0], \
                'M must be a square matrix of same size as K'

    def apply_constraint(self,
                         R: Union[npt.NDArray, scipy.sparse.csr_matrix],
                         T: Optional[Union[npt.NDArray, scipy.sparse.csr_matrix]] = None,
                         local: bool = True, epsilon: float = 1e-8):
        """
        Apply the constraint

        .. math::
            R \\, y = 0, \\qquad y = T z

        to the current or the global ``Stiffness`` object coordinate

        :param R: the constraint coefficient matrix
        :param T: the allowed node displacements; if ``None`` constraint is normalized
        :param local: if ``True`` the constraint is applied
        :param epsilon: precision used to assess numerical rank
        :return:

        **Notes:**

        1. If ``local = True`` and the current constraints are

           .. math::
               R_y \\, x = 0, \\quad x = T_y \\, y, \\qquad R^{}_y R_y^T = I, \\quad T_y^T T^{}_y = I, \\quad R_y T_y = 0

           then after applying the new constraints

           .. math::
               R \\, y = R \\, T_y^T x = R_z x = 0, \\qquad x = T_y \\, y = T_y T z = T_z z,

           in which

           .. math::
               R_z = R \\, T_y^T, \\qquad T_z = T_y T

           and

           .. math::
               V = \\frac{1}{2} z^T K_z \\, z + \\frac{1}{2} \\ddot{z}^T M_z \\, \\ddot{z}, \\qquad R_z x = 0, \\quad x = R_z \\, z,

           in which

           .. math::
               K_z = T^T K T, \\qquad M_z = T^T M T

        2. If ``local = False`` and the constraints are interpreted as

           .. math::
               R \\, x = 0, \\quad x = T \\, z

           which is first converted to the local constraint

           .. math::
               R \\, x = R \\, T_y y = \\tilde{R} y = 0, \\qquad \\tilde{R} = R \\, T_y,

           before applying
        """

        if local or self.T is None:
            # local constraint
            if T is None:
                # normalize before applying
                rank, r, t = orthogonalize(R.transpose(), mode='complete', epsilon=epsilon)
            else:
                # test orthogonality
                assert scipy.linalg.norm(R @ T) < epsilon, 'R and T must be orthogonal'
                assert scipy.linalg.norm(R @ R.transpose() - np.eye(R.shape[0])) < epsilon, 'R must be unitary'
                assert scipy.linalg.norm(T.transpose() @ T - np.eye(T.shape[1])) < epsilon, 'T must be unitary'
                r, t = R, T

            # apply constraint

            if scipy.sparse.issparse(self.K):
                # make sure it is sparse
                if scipy.sparse.issparse(r):
                    r = scipy.sparse.csr_matrix(r)
                if scipy.sparse.issparse(t):
                    t = scipy.sparse.csr_matrix(t)

            # project stiffness
            self.K = t.transpose() @ self.K @ t

            # symmetrize
            self.K += self.K.transpose()
            self.K /= 2

            if self.M is not None:
                # project mass
                self.M = t.transpose() @ self.M @ t
                self.M += self.M.transpose()
                self.M /= 2

            if self.R is None:
                self.R = r
            else:
                if self.T is None:
                    self.R = r
                else:
                    self.R = r @ self.T.transpose()

            if self.T is None:
                self.T = t
            else:
                self.T = self.T @ t

        else:
            # not local and self.T is not None
            if T is not None:
                warnings.warn("allowed displacements parameter 'T' ignored when 'local=False'")
            self.apply_constraint(R @ self.T, epsilon=epsilon)

    def displacements(self, f: npt.NDArray):
        """
        Calculate displacements due to the application of a global force

        :param f: 3 x m array of forces
        :return: the displacements

        **Notes:**

        1. The displacements are calculated in global coordinates

           .. math::
               x = T K^{-1} T^T f
        """
        m = self.K.shape[0] if self.T is None else self.T.shape[0]
        assert f.shape[0] == 3 and f.shape[1] == m/3, 'f must be a 3 x m array'
        x = self.T @ np.linalg.solve(self.K, self.T.transpose() @ f.flatten(order='F')) \
            if self.T is not None else \
            np.linalg.solve(self.K, f.flatten(order='F'))
        return x.reshape((3, m/3))

    def eigs(self, k: int = 12, which: Literal['LM', 'SM', 'LR', 'SR', 'LI', 'SI'] = 'SM'):
        """
        Compute the eigenvalues and eigenvectors of the stiffness matrix

        If the stiffness matrix is stored as a sparse array, return only the first k eigenvalue/eigenvector pairs

        :param k: the number of eigenvalues
        :param which: which eigenvalues to compute; see :func:`scipy.sparse.linalg.eigsh` for details
        :return: tuple with eigenvalues, and eigenvectors

        **Notes:**

        1. The eigenvalues and eigenvectors are calculated by solving the eigenvalue problem

           .. math::
               K y = \\lambda \\, y, \\qquad x = T \\, y
        """

        if scipy.sparse.issparse(self.K):
            # calculate sparse eigenvalues
            d, v = scipy.sparse.linalg.eigsh(self.K, k=k, which=which)

        else:
            # calculate dense eigenvalues
            d, v = np.linalg.eigh(self.K)

        # sort
        ind = np.argsort(d)

        # project eigenvectors
        if self.T is not None:
            v = self.T @ v

        return d[ind], v[:, ind]

    def modes(self, k: int = 12, units='Hz', which: Literal['LM', 'SM'] = 'SM'):
        """
        Compute the natural frequencies [rad/s] and mode vectors

        :param k: the number of eigenvalues
        :param which: which eigenvalues to compute; see :func:`scipy.sparse.linalg.eigsh` for details
        :param units: return frequencies in Hz if ``units = 'Hz'``
        :return: tuple with eigenvalues, and eigenvectors

        **Notes:**

        1. The natural frequencies and mode vectors are calculated by solving the generalized eigenvalue problem

           .. math::
               K y = \\omega^2 M y, \\qquad x = T \\, y
        """

        assert self.M is not None, 'modes cannot be calculated without mass information'

        if scipy.sparse.issparse(self.K):
            # calculate sparse eigenvalues
            d, v = scipy.sparse.linalg.eigsh(self.K, k=k, M=self.M, which=which)
        else:
            # calculate dense eigenvalues
            d, v = scipy.linalg.eigh(self.K, self.M)

        # calculate frequencies
        d = np.sqrt(np.abs(d))
        if units == 'Hz':
            d /= 2*np.pi

        # sort
        ind = np.argsort(d)

        # project eigenvectors
        if self.T is not None:
            v = self.T @ v

        return d[ind], v[:, ind]

