import numpy as np
import scipy
from scipy.special import expit

class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)
    


class BarrierLassoOracle(BaseSmoothOracle):
    """
    Oracle for log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.
    """
    def __init__(self, A, b, regcoef, t):
        self.A = A
        self.matvec_Ax = lambda x: A @ x
        self.matvec_ATx = lambda x: A.T @ x
        self.b = b
        self.regcoef = regcoef
        self.t = t

    def lasso_func(self, x, u):
        return np.linalg.norm(self.matvec_Ax(x) - self.b) ** 2 / 2 + self.regcoef * np.sum(u)

    def func(self, x):
        x, u = np.array_split(x, 2)
        return self.t * self.lasso_func(x, u) - np.sum(np.log(u + x) + np.log(u - x))

    def grad(self, x):
        x, u = np.array_split(x, 2)
        grad_f_x = self.t * self.matvec_ATx(self.matvec_Ax(x) - self.b) - 1 / (u + x) + 1 / (u - x)
        grad_f_u = self.t * self.regcoef * np.ones(len(u)) - 1 / (u + x) - 1 / (u - x)
        return np.hstack([grad_f_x, grad_f_u])

    def hess(self, x):
        x, u = np.array_split(x, 2)
        hess_xx = self.A.T @ self.A * self.t + np.diag(1 / (u - x) ** 2 + 1 / (u + x) ** 2)
        hess_xu = np.diag(1 / (u + x) ** 2 - 1 / (u - x) ** 2)
        hess_uu = np.diag(1 / (u + x) ** 2 + 1 / (u - x) ** 2)
        return np.vstack((np.hstack((hess_xx, hess_xu)), np.hstack((hess_xu, hess_uu))))


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    mu = np.min([1., regcoef /np.linalg.norm(ATAx_b, np.inf)]) * Ax_b

    return np.linalg.norm(0.5 * Ax_b)**2 + regcoef * np.linalg.norm(x, 1) + 0.5 * np.linalg.norm(mu, 2)**2 + np.dot(b, mu)
