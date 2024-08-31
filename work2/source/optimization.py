import time
import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool


def debug_info(iteration, x_k, oracle_at_x_k):
  print('Iteration:', iteration)
  print('x_k:', x_k, '\toracle_at_x_k:', oracle_at_x_k)
  print()

def update_history(history, time_start, func, grad_norm, x):
  history['func'].append(func)
  history['time'].append(time.time() - time_start)
  history['grad_norm'].append(grad_norm)
  if x.size <= 2:
    history['x'].append(x)
  return history


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    if trace:
       history = defaultdict(list)
       time_start = time.time()
    else:
       history =  None
       time_start = None
    if max_iter is None:
       max_iter = x_0.size

    x_k = np.copy(x_0)
    g_k = matvec(x_k) - b
    d_k = - g_k
    Ad_k = matvec(d_k)
    
    status = False
    for k in range(max_iter):
        alpha_k = g_k.dot(g_k) / (d_k.dot(Ad_k))
        x_k = x_k + alpha_k * d_k
        g_k_plus_1 = g_k + alpha_k * Ad_k
        residual_norm = np.linalg.norm(g_k_plus_1, 2)
        
        if display:
            print('Iteration:', k + 1)
            print('x_k:', x_k, '\tresidual_norm:', residual_norm)
            print()
        if trace:
            history['time'].append(time.time() - time_start)
            history['residual_norm'].append(residual_norm)
            if x_k.size <= 2:
                history['x'].append(x_k)  

        if residual_norm**2 <= tolerance:
            status = True
            break
        else:
           beta_k = g_k_plus_1.dot(g_k_plus_1) / (g_k.dot(g_k))
           d_k = - g_k_plus_1 + beta_k * d_k
           g_k = np.copy(g_k_plus_1)
           Ad_k = matvec(d_k)     

    if not status:
      return x_k, 'iterations_exceeded', history

    return x_k, 'success', history



def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    ################################################

    def lbfgs_direction(grad, h):
        res = -grad
        if len(h) > 0:
            mus = []
            for s, y in reversed(h):
                mu = np.dot(s, res) / np.dot(s, y)
                mus.append(mu)
                res -= mu * y

            s_last, y_last = h[-1]
            res *= np.dot(s_last, y_last) / np.dot(y_last, y_last)

            for (s, y), mu in zip(h, reversed(mus)):
                beta = np.dot(y, res) / np.dot(s, y)
                res += (mu - beta) * s
        return res       

    ################################################

    if trace:
        history = defaultdict(list)
        time_start = time.time()
    else:
        history =  None
        time_start = None

    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    grad_norm_x0 = np.linalg.norm(oracle.grad(x_k))
    h = deque(maxlen=memory_size)

    status = False
    for k in range(max_iter):
        oracle_at_x_k = oracle.func(x_k)
        grad_x_k = oracle.grad(x_k)
        d_k = lbfgs_direction(grad_x_k, h)
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, 1.0) 
        x_k_1 = x_k + alpha_k * d_k
        
        s_k = x_k_1 - x_k
        grad_x_k_1 = oracle.grad(x_k_1)
        y_k = grad_x_k_1 - grad_x_k
        h.append((s_k, y_k))
        grad_x_k = grad_x_k_1
        x_k = x_k_1
        grad_norm = np.linalg.norm(grad_x_k, 2)

        if trace: update_history(history, time_start, oracle_at_x_k, grad_norm, x_k)
        if display: debug_info(k + 1, x_k, oracle_at_x_k)

        if grad_norm**2 <= tolerance * grad_norm_x0**2:
            status = True
            break

    if not status:
        return x_k, 'iterations_exceeded', history

    return x_k, 'success', history



def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    if trace:
        history = defaultdict(list)
        time_start = time.time()
    else:
       history =  None
       time_start = None

    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    g_k = oracle.grad(x_k)
    d_k = - g_k
    g_k_norm = np.linalg.norm(g_k, 2)

    status = False
    for k in range(max_iter):
        epsilon_k = np.min([0.5, np.sqrt(g_k_norm)]) * g_k_norm
        d_k, _, _ = conjugate_gradients(lambda v: oracle.hess_vec(x_k, v), -g_k, d_k, tolerance=epsilon_k)
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, 1.0)
        x_k = x_k + alpha_k * d_k
        g_k = oracle.grad(x_k)
        g_k_norm = np.linalg.norm(g_k, 2)
        oracle_at_x_k = oracle.func(x_k)

        if display: debug_info(k + 1, x_k, oracle_at_x_k)
        if trace: update_history(history, time_start, oracle_at_x_k, g_k_norm, x_k)

        if g_k_norm**2 <= tolerance:
            status = True
            break

    if not status:
      return x_k, 'iterations_exceeded', history

    return x_k, 'success', history



def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
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
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
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
      grad_norm_at_x_k = np.linalg.norm(grad_at_x_k)

      if display: debug_info(k + 1, x_k, oracle_at_x_k)
      if trace: update_history(history, time_start, oracle_at_x_k, grad_norm_at_x_k, x_k)

      if grad_norm_at_x_k**2 <= tolerance * grad_norm_at_x_0**2:
        status = True
        break
      else:
        d_k = (-1) * grad_at_x_k
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k)
        x_k = x_k + alpha_k * d_k

    if not status:
      return x_k, 'iterations_exceeded', history

    return x_k, 'success', history
