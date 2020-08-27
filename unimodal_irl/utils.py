"""Various utility methods related to IRL algorithms"""

import gym
import copy
import warnings
import numpy as np
import itertools as it

from scipy.optimize import OptimizeResult

from explicit_env.envs.utils import compute_parents_children


def get_rollouts(env, policy, num_rollouts, *, max_episode_length=None, verbose=False):
    """Get rollouts of a policy in an environment
    
    Args:
        env (gym.Env): Environment to use
        policy (object): Policy object with a .predict() method matching the
            stable-baselines policy API.
        num_rollouts: Number of rollouts to collect
        
        max_episode_length (int): If provided, stop trajectories at this length
        verbose (bool): If true, print extra logging info
        
    Returns:
        (list): List of [(s, a), (s, a), ..., (s, None)] trajectories
    """

    rollouts = []
    for episode in it.count():

        # Prepare one trajectory
        rollout = []

        # Reset environment
        s = env.reset()
        for timestep in it.count():

            # Sample action from policy
            a, _ = policy.predict(s)

            # Add state, action to trajectory
            rollout.append((s, a))

            # Step environment
            s, r, done, _ = env.step(a)

            if done:
                break

            if max_episode_length is not None:
                if timestep == max_episode_length - 2:
                    if verbose:
                        print("Stopping after reaching maximum episode length")
                    break

        rollout.append((s, None))
        rollouts.append(rollout)

        if episode == num_rollouts - 1:
            break

    return rollouts


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


def empirical_feature_expectations(env, rollouts):
    """Find empirical discounted feature expectations
    
    Args:
        env (unimodal_irl.envs.explicit.IExplicitEnv): Environment defining dynamics,
            reward(s) and discount factor
        rollouts (list): List of [(s, a), (s, a), ..., (s, None)] trajectories
    
    Returns:
        (numpy array): |S| array of state marginals
        (numpy array): |S|x|A| array of state-action marginals
        (numpy array): |S|x|A|x|S| array of state-action-state marginals
    """

    # Find discounted feature expectations
    phibar_s = np.zeros(env.t_mat.shape[0])
    phibar_sa = np.zeros(env.t_mat.shape[0 : 1 + 1])
    phibar_sas = np.zeros(env.t_mat.shape)
    for r in rollouts:

        if env.state_rewards is not None:
            for t, (s1, _) in enumerate(r):
                phibar_s[s1] += (env.gamma ** t) * 1

        if env.state_action_rewards is not None:
            for t, (s1, a) in enumerate(r[:-1]):
                phibar_sa[s1, a] += (env.gamma ** t) * 1

        if env.state_action_state_rewards is not None:
            for t, (s1, a) in enumerate(r[:-1]):
                s2 = r[t + 1][0]
                phibar_sas[s1, a, s2] += (env.gamma ** t) * 1

    # Divide by # demonstrations to get means
    norm = 1 / len(rollouts)
    phibar_s *= norm
    phibar_sa *= norm
    phibar_sas *= norm

    return phibar_s, phibar_sa, phibar_sas


def minimize_vgd(f, x0, args=(), bounds=None, **kwargs):
    """Minimize a function using vanilla gradient descent
    
    Args:
        f (function): Objective function taking x0, args and returning objective value
            and gradient vector
        x0 (numpy array): Initial parameter guess
        args (tuple): Optional collection of extra arguments for obj()
        bounds (iterable): List of (min, max) bounds for each variable in x0
    
    Returns:
    
    """
    x = x0
    step_size = 0.1
    tol = 1e-3

    nit = 1
    nfev = 1
    status = ""
    message = ""
    for iter in it.count():
        fun, jac = f(x, *args)
        x_new = x - step_size * jac

        # Clip to bounds
        x_new = np.clip(x_new, [mn for (mn, mx) in bounds], [mx for (mn, mx) in bounds])

        delta = np.max(np.abs(x_new - x))
        x = x_new

        if delta <= tol:
            status = "converged"
            message = f"max(abs(gradient)) <= desired tolerance ({tol})"
            break

        nit += 1
        nfev += 1

    return OptimizeResult(
        {
            "x": x,
            "success": True,
            "status": status,
            "message": message,
            "fun": fun,
            "jac": jac,
            "nfev": nfev,
            "nit": nit,
        }
    )
