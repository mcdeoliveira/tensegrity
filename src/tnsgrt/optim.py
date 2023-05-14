import cvxpy as cvx
import numpy as np


def qp(n, m, Q, c, A, blo, bup, xlo, xup, **kwargs):
    """
    Minimize 1/2 x^T Q x + 1/2 f^T D f + c^T x
      s.t.   blo <= A x <= bup
             xlo <= x <= xup
             f = F^T x
    kwargs:
     - F and D
    """

    # variable
    x = cvx.Variable(n)

    # factored cost
    F = kwargs.pop('F', None)
    D = kwargs.pop('D', None)

    # objective function
    objective = 0
    if Q is not None:
        objective += (1 / 2) * cvx.quad_form(x, Q)
    if F is not None and D is not None:
        objective += (1 / 2) * cvx.quad_form(F.T @ x, D)
    if c is not None:
        objective += c.T @ x
    objective = cvx.Minimize(objective)

    # constraints
    constraints = []

    # variable upper bounds
    if xup is not None:
        isconstrained = np.logical_not(np.isinf(xup))
        constraints.append(x[isconstrained] <= xup[isconstrained])

    # variable lower bounds
    if xlo is not None:
        isconstrained = np.logical_not(np.isneginf(xlo))
        constraints.append(x[isconstrained] >= xlo[isconstrained])

    # constraints
    if m > 0:
        # upper and lower bounds
        if bup is not None and blo is not None:
            # equality constraints?
            isequal = blo == bup
            if np.any(isequal):
                constraints.append(A[isequal, :] @ x == bup[isequal])

            # upper bound constraints?
            isconstrained = np.logical_not(np.logical_or(np.isinf(bup), isequal))
            if np.any(isconstrained):
                constraints.append(A[isconstrained, :] @ x <= bup[isconstrained])

            isconstrained = np.logical_not(np.logical_or(np.isneginf(blo), isequal))
            if np.any(isconstrained):
                constraints.append(A[isconstrained, :] @ x >= blo[isconstrained])

        else:

            # a upper bounds
            if bup is not None:
                isconstrained = np.logical_not(np.isinf(bup))
                constraints.append(A[isconstrained, :] @ x <= bup[isconstrained])

            # a lower bounds
            if blo is not None:
                isconstrained = np.logical_not(np.isneginf(blo))
                constraints.append(A[isconstrained, :] @ x >= blo[isconstrained])

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


def lp(n, m, c, A, blo, bup, xlo, xup, **kwargs):
    """
    Minimize c^T x
      s.t.   blo <= A x <= bup
             xlo <= x <= xup
    """
    return qp(n, m, None, c, A, blo, bup, xlo, xup, **kwargs)
