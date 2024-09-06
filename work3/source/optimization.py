import time
from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve, LinAlgError
import scipy

import oracles
from utils import get_line_search_tool    


def debug_info(iteration, x_k, oracle_at_x_k):
  print('Iteration:', iteration)
  print('x_k:', x_k, '\toracle_at_x_k:', oracle_at_x_k)
  print()

def update_history(history, time_start, x, func, grad_norm=None, duality_gap=None):
  history['time'].append(time.time() - time_start)
  history['func'].append(func)
  history['grad_norm'].append(grad_norm)
  history['duality_gap'].append(duality_gap)
  if x.size <= 2:
    history['x'].append(x)
  return history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """

    if trace:
      history = defaultdict(list)
      time_start = time.time()
    else:
      history =  None
      time_start = None


    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    grad_norm_at_x_0 = np.linalg.norm(oracle.grad(x_0))
    status = False

    for k in range(max_iter):
        if (~np.isfinite(x_k)).any():
            return x_k, 'computational_error', history
        
        oracle_at_x_k = oracle.func(x_k)
        grad_at_x_k = oracle.grad(x_k)
        hess_at_x_k = oracle.hess(x_k)
        grad_norm_at_x_k = np.linalg.norm(grad_at_x_k)

        if display: debug_info(k + 1, x_k, oracle_at_x_k)
        if trace: update_history(history, time_start, x_k, oracle_at_x_k, grad_norm=grad_norm_at_x_k)
        
        if grad_norm_at_x_k**2 <= tolerance * grad_norm_at_x_0**2:
            status = True
            break
        else:
            try:
                c, low = scipy.linalg.cho_factor(hess_at_x_k)
                d_k = scipy.linalg.cho_solve((c, low), (-1) * grad_at_x_k)
            except np.linalg.LinAlgError:
                return x_k, 'newton_direction_error', history
            
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        x_k = x_k + alpha_k * d_k        
  
    if not status:
        x_k, 'iterations_exceeded', history

    return x_k, 'success', history



def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5, 
                         tolerance_inner=1e-8, max_iter=100, 
                         max_iter_inner=20, t_0=1, gamma=10, 
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary 
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    if trace:
      history = defaultdict(list)
      time_start = time.time()
    else:
      history =  None
      time_start = None


    oracle = oracles.BarrierLassoOracle(A, b, reg_coef, t_0)
    x_k, u_k = np.copy(x_0), np.copy(u_0)
    xu_k = np.concatenate([x_k, u_k])
    t_k = t_0
    oracle_at_x_k = oracle.lasso_func(x_k, u_k)
    status = False

    lasso_duality_gap = lasso_duality_gap if lasso_duality_gap else oracles.lasso_duality_gap
    lasso_duality_gap_ = lambda x_: lasso_duality_gap(x_, A @ x_ - b, A.T @ (A @ x_ - b), b, reg_coef)
    duality_gap_k = lasso_duality_gap_(x_k)

    for k in range(max_iter):
        if display: debug_info(k + 1, x_k, oracle_at_x_k)
        if trace: update_history(history, time_start, x_k, oracle_at_x_k, duality_gap=duality_gap_k)

        if duality_gap_k < tolerance:
            status = True
            break

        xu_k, message, _ = newton(oracle, xu_k, max_iter=max_iter_inner, tolerance=tolerance_inner,
                                  line_search_options={'method': 'Armijo', 'c1' : c1})
        if message == 'computational_error':
            return (x_k, u_k), 'computational_error', history

        x_k, u_k = np.array_split(xu_k, 2)
        t_k *= gamma
        oracle.t = t_k
        oracle_at_x_k = oracle.lasso_func(x_k, u_k)
        duality_gap_k = lasso_duality_gap_(x_k)

    if not status:
        return (x_k, u_k), 'iterations_exceeded', history

    return (x_k, u_k), 'success', history
