"""Various utility methods related to IRL algorithms"""

import numpy as np
import itertools as it

from scipy.optimize import OptimizeResult


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
