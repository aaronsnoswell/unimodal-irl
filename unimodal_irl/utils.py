"""Various utility methods related to IRL algorithms"""

import gym
import copy
import warnings
import numpy as np
import itertools as it

from scipy.optimize import OptimizeResult

from explicit_env.envs.utils import compute_parents_children


def pad_terminal_mdp(env, *, rollouts=None, max_length=None):
    """Pads a terminal MDP, adding a dummy state and action
    
    We gain a O(|S|) space and time efficiency improvement with our MaxEnt IRL algorithm
    for MDPs with terminal states by transforming them to have no terminal states. This
    is done by adding a dummy state and action that pad any trajectories out to a fixed
    upper length.
    
    Args:
        env (.explicit.IExplicitEnv) Explicit MDP environment
        
        rollouts (list): List of [(s, a), (s, a), ..., (s, None)] rollouts to pad
        max_length (int): Optional maximum length to pad to, otherwise paths are padded
            to match the length of the longest path
    
    Returns:
        (.explicit.IExplicitEnv) Explicit MDP environment, padded with a dummy
            state and action so that it has no terminal states.
        (list): List of rollouts, padded to max_length
    """

    env = copy.deepcopy(env)

    # Add an extra state and action to the dynamics
    t_mat2 = np.pad(env.t_mat, (0, 1), mode="constant")
    p0s2 = np.pad(env.p0s, (0, 1), mode="constant")
    terminal_state_mask2 = np.pad(env.terminal_state_mask, (0, 1), mode="constant")
    states2 = np.arange(t_mat2.shape[0])
    actions2 = np.arange(t_mat2.shape[1])

    state_rewards2 = None
    state_action_rewards2 = None
    state_action_state_rewards2 = None

    # Dummy state is absorbing
    t_mat2[-1, -1, -1] = 1

    # Terminal states are no longer absorbing
    for terminal_state in np.argwhere(env.terminal_state_mask):
        t_mat2[terminal_state, :, terminal_state] = 0
    terminal_state_mask2 = np.zeros(t_mat2.shape[0])

    # Dummy state reachable anywhere if dummy action is taken
    t_mat2[:, -1, -1] = 1

    # Dummy state doesn't modify rewards
    if env.state_rewards is not None:
        state_rewards2 = np.pad(env.state_rewards, (0, 1), mode="constant")
        state_rewards2[-1] = 0
    if env.state_action_rewards is not None:
        state_action_rewards2 = np.pad(
            env.state_action_rewards, (0, 1), mode="constant"
        )
        state_action_rewards2[:, -1] = 0
    if env.state_action_state_rewards is not None:
        state_action_state_rewards2 = np.pad(
            env.state_action_state_rewards, (0, 1), mode="constant"
        )
        state_action_state_rewards2[:, 0:-1, -1] = -np.inf  # Illegal transition
        state_action_state_rewards2[:, -1, -1] = 0

    # Overwrite environment properties
    env._states = states2
    env._actions = actions2
    env._t_mat = t_mat2
    env._p0s = p0s2
    env._terminal_state_mask = terminal_state_mask2
    env.observation_space = gym.spaces.Discrete(env.observation_space.n + 1)
    env.action_space = gym.spaces.Discrete(env.action_space.n + 1)

    # Update parent and children mappings
    env._parents, env._children = compute_parents_children(
        env.t_mat, env.terminal_state_mask
    )

    if env.state_rewards is not None:
        env._state_rewards = state_rewards2
    if env.state_action_rewards is not None:
        env._state_action_rewards = state_action_rewards2
    if env.state_action_state_rewards is not None:
        env._state_action_state_rewards = state_action_state_rewards2

    # Finally, pad the trajectories
    if rollouts is None:
        return env
    else:
        # Measure the length of the rollouts
        r_len = [len(r) for r in rollouts]
        if max_length is None:
            max_length = max(r_len)
        elif max_length < max(r_len):
            warnings.warn(
                f"Provided max length ({max_length}) is < maximum path length ({max(r_len)}), using maximum path length instead"
            )
            max_length = max(r_len)

        _rollouts = []
        dummy_state = t_mat2.shape[0] - 1
        dummy_action = t_mat2.shape[1] - 1
        for r in rollouts:
            _r = r.copy()
            if len(_r) < max_length:
                s, _ = _r[-1]
                _r[-1] = (s, dummy_action)
                while len(_r) != max_length - 1:
                    _r.append((dummy_state, dummy_action))
                _r.append((dummy_state, None))
            _rollouts.append(_r)
        return env, _rollouts


def empirical_feature_expectations(env, rollouts, weights=None):
    """Find empirical discounted feature expectations
    
    Args:
        env (unimodal_irl.envs.explicit.IExplicitEnv): Environment defining dynamics
            and discount factor
        rollouts (list): List of [(s, a), (s, a), ..., (s, None)] trajectories
        weights (numpy array): Optional list of weights that can augment the path
            feature expectations. Defaults to uniform.
    
    Returns:
        (numpy array): |S| array of state marginals
        (numpy array): |S|x|A| array of state-action marginals
        (numpy array): |S|x|A|x|S| array of state-action-state marginals
    """

    if weights is None:
        # Default to uniform path weighting
        weights = np.ones(len(rollouts))
    else:
        assert len(weights) == len(
            rollouts
        ), f"Path weights are not correct size, should be {len(rollouts)}, are {len(weights)}"

    # Find discounted feature expectations
    phibar_s = np.zeros(env.t_mat.shape[0])
    phibar_sa = np.zeros(env.t_mat.shape[0 : 1 + 1])
    phibar_sas = np.zeros(env.t_mat.shape)
    for r, w in zip(rollouts, weights):

        for t, (s1, _) in enumerate(r):
            phibar_s[s1] += w * (env.gamma ** t)

        for t, (s1, a) in enumerate(r[:-1]):
            phibar_sa[s1, a] += w * (env.gamma ** t)

            s2 = r[t + 1][0]
            phibar_sas[s1, a, s2] += w * (env.gamma ** t)

    phibar_s /= np.sum(weights)
    phibar_sa /= np.sum(weights)
    phibar_sas /= np.sum(weights)

    return phibar_s, phibar_sa, phibar_sas


def minimize_vgd(
    f,
    x0,
    args=(),
    bounds=None,
    options=dict(disp=False, maxiter=None, step_size=1e-5, tol=1e-3),
    **kwargs,
):
    """Minimize a function using vanilla gradient descent
    
    TODO ajs 28/Oct/2020 integrate optimal step-size selection via Wolfe line search
    C.f. https://github.com/tttor/hiord-opt/blob/master/tor_opt/wolfe_strong.py
    C.f. https://github.com/tttor/hiord-opt/blob/master/tor_opt/lbfgs.py
    C.f. https://github.com/tttor/hiord-opt/blob/master/tor_opt/test/test_opt.py
    
    Args:
        f (function): Objective function taking x0, args and returning objective value
            and gradient vector
        x0 (numpy array): Initial parameter guess
        
        args (tuple): Optional collection of extra arguments for obj()
        bounds (iterable): List of (min, max) bounds for each variable in x0
        options (dict): Option dictionary. Valid entries are;
            disp (bool): Display logging info
            maxiter (int): Maximum number of iterations to perform
            step_size (float or callable): Step size to use when taking gradient steps.
                Can be a float, or a callable. If a callable, it should accept three
                arguments; the iteration count (zero based), the current point, and the
                current jacobian vector. If not provided, defaults to 1e-5.
            tol (float): Convergence tolerance. VGD will terminate when the L2 change in
                iterates falls below this value.
    
    Returns:
        (scipy.optimize.OptimizeResult): Result object
    """

    disp = options.get("disp", False)
    maxiter = options.get("maxiter", None)
    tol = options.get("tol", 1e-3)
    step_size = options.get("step_size", 1e-5)

    x = x0
    x_best = x.copy()
    fun_best = np.inf

    nit = 1
    nfev = 1
    status = ""
    message = ""
    for iter in it.count():
        fun, jac = f(x, *args)

        if fun < fun_best:
            x_best = x.copy()
            fun_best = fun

        if callable(step_size):
            _step_size = step_size(iter, x, jac)
        else:
            _step_size = step_size
        x_new = x - _step_size * jac

        if bounds is not None:
            # Clip to bounds
            x_new_clipped = np.clip(
                x_new, [mn for (mn, mx) in bounds], [mx for (mn, mx) in bounds]
            )
        else:
            x_new_clipped = x_new

        delta = np.max(np.abs(x_new_clipped - x))

        if disp:
            # print(fun)
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

        if maxiter is not None:
            if iter == maxiter:
                if disp:
                    print("Maximum number of iterations reached")
                success = False
                status = 2
                message = f"Maximum number of iterations ({maxiter}) reached"
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
        }
    )

    if disp:
        print(res)

    return res
