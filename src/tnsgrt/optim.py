from typing import Optional, Union, Tuple

import cvxpy as cvx
import numpy as np
import numpy.typing as npt
import scipy


def infer_dimensions(c: Optional[npt.NDArray],
                     a: Union[npt.NDArray, scipy.sparse.csr_array],
                     xlo: npt.NDArray, xup: npt.NDArray) -> Tuple[int, int]:
    m: int = 0
    if a is not None:
        m: int = a.shape[0]
        n: int = a.shape[1]
    elif c is not None:
        n: int = c.shape[0]
    elif xlo is not None or xup is not None:
        if xlo is not None:
            n: int = xlo.shape[0]
        else:
            n: int = xup.shape[0]
    else:
        raise Exception('cannot infer problem size')
    return m, n


def assert_dimensions(label, value, shape):
    if value is not None:
        assert value.ndim == len(shape), f'{label} must be {len(shape)} dimensional'
        assert value.shape == shape, 'f{label} must be a {shape} dimensional array'


def qp(Q: Optional[Union[npt.NDArray, scipy.sparse.csr_array]],
       c: npt.NDArray,
       A: Union[npt.NDArray, scipy.sparse.csr_array],
       blo: npt.NDArray, bup: npt.NDArray,
       xlo: npt.NDArray, xup: npt.NDArray, **kwargs) -> Tuple[float, npt.NDArray, str]:
    """
    Minimize 1/2 x^T Q x + 1/2 f^T D f + c^T x
      s.t.   blo <= A x <= bup
             xlo <= x <= xup
             f = F^T x
    kwargs:
     - F and D
    """

    # check dimensions
    m, n = infer_dimensions(c, A, xlo, xup)
    assert_dimensions('A', A, (m, n))
    assert_dimensions('blo', blo, (m,))
    assert_dimensions('bup', bup, (m,))
    assert_dimensions('Q', Q, (m, n))
    assert_dimensions('c', c, (n,))
    assert_dimensions('xlo', xlo, (n,))
    assert_dimensions('xup', xup, (n,))

    # variable
    x = cvx.Variable(n)

    # factored cost
    F = kwargs.pop('F', None)
    D = kwargs.pop('D', None)

    # objective function
    obj: cvx.Expression = 0
    if Q is not None:
        obj += (1 / 2) * cvx.quad_form(x, Q)
    if F is not None and D is not None:
        obj += (1 / 2) * cvx.quad_form(F.T @ x, D)
    if c is not None:
        obj += c.T @ x
    objective = cvx.Minimize(obj)

    # constraints
    constraints = []

    # variable upper bounds
    if xup is not None:
        is_constrained = np.logical_not(np.isinf(xup))
        constraints.append(x[is_constrained] <= xup[is_constrained])

    # variable lower bounds
    if xlo is not None:
        is_constrained = np.logical_not(np.isneginf(xlo))
        constraints.append(x[is_constrained] >= xlo[is_constrained])

    # constraints
    if m > 0:
        # upper and lower bounds
        if bup is not None and blo is not None:
            # equality constraints?
            isequal = blo == bup
            if np.any(isequal):
                constraints.append(A[isequal, :] @ x == bup[isequal])

            # upper bound constraints?
            is_constrained = np.logical_not(np.logical_or(np.isinf(bup), isequal))
            if np.any(is_constrained):
                constraints.append(A[is_constrained, :] @ x <= bup[is_constrained])

            is_constrained = np.logical_not(np.logical_or(np.isneginf(blo), isequal))
            if np.any(is_constrained):
                constraints.append(A[is_constrained, :] @ x >= blo[is_constrained])

        else:

            # a upper bounds
            if bup is not None:
                is_constrained = np.logical_not(np.isinf(bup))
                constraints.append(A[is_constrained, :] @ x <= bup[is_constrained])

            # a lower bounds
            if blo is not None:
                is_constrained = np.logical_not(np.isneginf(blo))
                constraints.append(A[is_constrained, :] @ x >= blo[is_constrained])

    # problem
    problem = cvx.Problem(objective, constraints)

    try:

        # solve
        defaults = {
            # 'solver': cvx.OSQP,
            # 'eps_abs': 1e-8
        }
        defaults.update(**kwargs)
        problem.solve(**defaults)

    except cvx.error.SolverError:

        # solve using default solver
        problem.solve(eps_abs=1e-8, **kwargs)

    return objective.value, x.value, problem.status


def lp(c, A, blo, bup, xlo, xup, **kwargs) -> Tuple[float, npt.NDArray, str]:
    """
    Minimize c^T x
      s.t.   blo <= A x <= bup
             xlo <= x <= xup
    """
    return qp(None, c, A, blo, bup, xlo, xup, **kwargs)


def feasibility(A, blo, bup, xlo, xup, epsilon=1e-8) -> \
        Tuple[bool, npt.NDArray, npt.NDArray]:
    """
    Minimize l1 + l2
      s.t.   blo <= A x + l1 - l2 <= bup
             xlo <= x <= xup
    """
    if A is None:
        return np.all(xup >= xlo), xlo, np.array([])
    m, n = infer_dimensions(None, A, xlo, xup)
    if scipy.sparse.issparse(A):
        Ax = np.hstack((A, scipy.sparse.eye(m), -scipy.sparse.eye(m)))
    else:
        Ax = np.hstack((A, np.eye(m), -np.eye(m)))
    cx = np.hstack((np.zeros((n,)), np.ones((2*m,))))
    if xlo is None:
        xlox = np.hstack((np.full((n,), -np.inf), np.ones(2*m,)))
    else:
        xlox = np.hstack((xlo, np.zeros(2*m, )))
    if xup is not None:
        xupx = np.hstack((xup, np.full((2*m,), np.inf)))
    else:
        xupx = None
    # solve feasibility problem
    val, x, feas = lp(cx, Ax, blo, bup, xlox, xupx)
    ls = x[n:n+m] - x[n+m:]
    return np.abs(ls) < epsilon, x[:n], ls
