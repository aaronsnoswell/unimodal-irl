"""Implementation of Vanilla Gradient Descent

(c) 2020 Aaron Snoswell

MIT Licensed
"""


import numpy as np
import itertools as it
from scipy.optimize import OptimizeResult, line_search


def vgd(objective, x0, args=(), **kwargs):
    """Minimize a scalar function with vanilla gradient descent
    
    This function is intended to be used with scipy.optimize.minimize(), e.g.
    
    ```python
    from scipy.optimize import minimize
    
    def obj_func(x):
        obj = x[0] ** 2 + x[1] ** 2
        grad = [2 * x[0], 2 * x[1]]
        return obj, grad
    
    minimize(obj_func, [1.8, 1.7], method=vgd, tol=1e-5, jac=True, options=dict(step_size='ls'))
    ```
    
    This is not intended for production use, however this function is very useful for
    de-bugging optimization objectives!
    
    Args:
        objective (callable): Function that accepts a vector x as it's first argument,
            then anything in args. Should return the objective function value. Should
            have a property .derivative(x, args) that returns the jacobian vector for x.
            Such a function can be called by wrapping a regular objective function
            returning objective value AND jacobian vector) in scipy.optimize.MemoizeJac.
        x0 (numpy array): Starting point for minimization
        args (list): List of extra arguments to be passed to objective
        **kwargs (dict): Optional extra arguments.
            - bounds (list): List of (xmin, xmax) bounding values.
            - callback (callable): Optional function to call after each iteration.
                Should accept the current point, and return a boolean indicating if
                the optimization should terminate or not.
            - disp (bool): Display progress information
            - step_size (float or callable or 'ls'): Step size for gradient descent. If
                float, a fixed step size will be used (this will likely cause
                oscillation or divergence). If a callable, it should accept the current
                iteration (zero based), the current x point, and the current jacobian
                vector, and return a step size as a float. If the string 'ls',
                scipy.optimize.line_search() will be used to select a step size that
                satisfies the strong Wolfe conditions.
            - max_iterations (int): Maximum number of gradient descent steps to take.
            - tol (float): Terminate optimization when subsequent points differ in
                L1 norm by less than this amount.
        
    Returns:
        (scipy.optimize.OptimizeResult): Result of the optimization. This object will
            contain a few additional terms;
                - x_vals (list): List of attempted x values during optimization
                - fun_vals (list): List of objective function values during optimization
                - jac_vals (list): List of jacobian values during optimization
                - step_sizes (list): List of step sizes taken during optimization
                - iter_best (int): The iteration at which the best solution point was
                    taken from
            
        
    """

    # Read extra options
    bounds = kwargs.get("bounds", None)
    callback = kwargs.get("callback", None)
    disp = kwargs.get("disp", False)
    tol = kwargs.get("tol", 1e-5)

    # Read VGD specific options
    step_size = kwargs.get("step_size", "ls")
    max_iterations = kwargs.get("max_iterations", None)

    x = x0
    iter_best = 0
    x_best = x.copy()
    fun_best = np.inf

    nit = 1
    nfev = 1
    status = ""
    message = ""
    x_vals = []
    fun_vals = []
    jac_vals = []
    step_sizes = []
    for iter in it.count():

        fun = objective(x, *args)
        jac = np.array(objective.derivative(x, *args))

        if step_size == "ls":
            # Use line search to find step size that satisfies strong Wolfe conditions
            ls_fun = lambda x: objective(x, *args)
            ls_jac = lambda x: objective.derivative(x, *args)

            pk = -jac
            _step_size, fc, _, fun, _, _ = line_search(ls_fun, ls_jac, x, pk)

            if _step_size is None:
                # Line search failed
                if disp:
                    print("Line search failed")
                success = False
                status = 2
                message = f"Line search at f({x}) failed after {fc} iterations"
                break

            nfev += fc
            x_new = x + _step_size * pk

        else:

            if callable(step_size):
                _step_size = step_size(iter, x, jac)
            else:
                _step_size = step_size
            x_new = x - _step_size * jac

        x_vals.append(x)
        fun_vals.append(fun)
        jac_vals.append(jac)
        step_sizes.append(_step_size)

        if fun < fun_best:
            iter_best = iter
            x_best = x.copy()
            fun_best = fun

        if bounds is not None:
            # Clip to bounds
            x_new_clipped = np.clip(
                x_new, [mn for (mn, mx) in bounds], [mx for (mn, mx) in bounds]
            )
        else:
            x_new_clipped = x_new

        delta = np.max(np.abs(x_new_clipped - x))

        if disp:
            print(f"VGD Iteration t = {iter}, f(x) = {fun}")
            print(f"x = {x}")
            print(f"∇(x) = {jac}")
            print(f"α(t, x, ∇(x)) = {_step_size}")
            print(f"x' = {x_new_clipped}")
            if bounds is not None:
                at_bounds = 0
                for p, (mn, mx) in zip(x_new_clipped, bounds):
                    if p == mn or p == mx:
                        at_bounds += 1
                print(f"{at_bounds} parameter(s) are at the bounds")
            print(f"Δ(x) = {delta}")
            print()

        x = x_new_clipped

        if delta <= tol:
            if disp:
                print("Converged")
            success = True
            status = 0
            message = f"max(abs(gradient)) <= desired tolerance ({tol})"
            break

        if max_iterations is not None:
            if iter == max_iterations:
                if disp:
                    print("Maximum number of iterations reached")
                success = False
                status = 2
                message = f"Maximum number of iterations ({max_iterations}) reached"
                break

        if callback is not None:
            callback_status = callback(x)
            if callback_status:
                status = 2
                message = f"User callback requested termination"
                break

        nit += 1
        nfev += 1

    res = OptimizeResult(
        {
            "x": x_best,
            "success": success,
            "status": status,
            "message": message,
            "fun": fun,
            "jac": jac,
            "nfev": nfev,
            "nit": nit,
            "x_vals": x_vals,
            "fun_vals": fun_vals,
            "jac_vals": jac_vals,
            "iter_best": iter_best,
            "step_sizes": step_sizes,
        }
    )

    if disp:
        print(res)

    return res
