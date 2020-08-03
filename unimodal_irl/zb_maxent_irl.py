"""Implements Maximum Entropy IRL from Ziebart 2008, 2010"""

import copy
import numpy as np
import itertools as it

from numba import jit
from scipy.optimize import minimize

from unimodal_irl.sw_maxent_irl import backward_pass_log, partition_log
from unimodal_irl.utils import empirical_feature_expectations


def state_marginals_08(p0s, t_mat, rs, max_path_length):
    """Compute state marginals using Ziebart's 2008 algorithm
    
    This algorithm is designed to compute state marginals for an un-discounted MDP with
    state-feature reward basis functions.
    
    Args:
        p0s (numpy array): |S| array of state starting probabilities
        t_mat (numpy array): |S|x|A|x|S| array of transition probabilities
        rs (numpy array): |S| array of state reward weights
        max_path_length (int): Maximum path length to consider
        
    Returns:
        (numpy array): |S| array of un-discounted state marginals
    """

    # Prepare state and action partition arrays (Step 1)
    Z_s = np.ones(t_mat.shape[0])
    Z_a = np.zeros(t_mat.shape[1])

    # Compute local partition values (Step 2)
    for t in range(max_path_length):
        # Sweep actions
        Z_a[:] = 0
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    Z_a[a] += t_mat[s1, a, s2] * np.exp(rs[s1]) * Z_s[s2]

        # Sweep states
        Z_s[:] = 0
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    Z_s[s1] += Z_a[a]

    # Compute local action probabilities (Step 3)
    prob_sa = np.zeros(t_mat.shape[0 : 1 + 1])
    for s1 in range(t_mat.shape[0]):
        for a in range(t_mat.shape[1]):
            if Z_s[s1] == 0.0:
                # This state was never reached during the backward pass
                # Default to a uniform distribution over actions
                prob_sa[s1, :] = 1.0 / t_mat.shape[1]
            else:
                prob_sa[s1, a] = Z_a[a] / Z_s[s1]

    # Prepare state marginal array (Step 4)
    dst = np.zeros((t_mat.shape[0], max_path_length + 1))
    for t in range(max_path_length + 1):
        dst[:, t] = p0s

    # Compute state occurrences at each time (Step 5)
    for t in range(max_path_length):
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    dst[s1, t + 1] += dst[s2, t] * prob_sa[s1, a] * t_mat[s1, a, s2]

    # Sum frequencies to get state marginals (Step 6)
    ps = np.sum(dst, axis=1)
    return ps


@jit(nopython=True)
def nb_state_marginals_08(p0s, t_mat, rs, max_path_length):
    """Compute state marginals using Ziebart's 2008 algorithm
    
    This algorithm is designed to compute state marginals for an un-discounted MDP with
    state-feature reward basis functions.
    
    Args:
        p0s (numpy array): |S| array of state starting probabilities
        t_mat (numpy array): |S|x|A|x|S| array of transition probabilities
        rs (numpy array): |S| array of state reward weights
        max_path_length (int): Maximum path length to consider
        
    Returns:
        (numpy array): |S| array of un-discounted state marginals
    """

    # Prepare state and action partition arrays (Step 1)
    Z_s = np.ones(t_mat.shape[0])
    Z_a = np.zeros(t_mat.shape[1])

    # Compute local partition values (Step 2)
    for t in range(max_path_length):
        # Sweep actions
        Z_a[:] = 0
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    Z_a[a] += t_mat[s1, a, s2] * np.exp(rs[s1]) * Z_s[s2]

        # Sweep states
        Z_s[:] = 0
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    Z_s[s1] += Z_a[a]

    # Compute local action probabilities (Step 3)
    prob_sa = np.zeros(t_mat.shape[0 : 1 + 1])
    for s1 in range(t_mat.shape[0]):
        for a in range(t_mat.shape[1]):
            if Z_s[s1] == 0.0:
                # This state was never reached during the backward pass
                # Default to a uniform distribution over actions
                prob_sa[s1, :] = 1.0 / t_mat.shape[1]
            else:
                prob_sa[s1, a] = Z_a[a] / Z_s[s1]

    # Prepare state marginal array (Step 4)
    dst = np.zeros((t_mat.shape[0], max_path_length + 1))
    for t in range(max_path_length + 1):
        dst[:, t] = p0s

    # Compute state occurrences at each time (Step 5)
    for t in range(max_path_length):
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    dst[s1, t + 1] += dst[s2, t] * prob_sa[s1, a] * t_mat[s1, a, s2]

    # Sum frequencies to get state marginals (Step 6)
    ps = np.sum(dst, axis=1)
    return ps


def state_marginals_10(p0s, t_mat, terminal_state_mask, rs, max_path_length):
    """Compute state marginals using Ziebart's 2010 algorithm
    
    This algorithm is designed to compute state marginals for an un-discounted MDP with
    state-feature reward basis functions.
    
    Args:
        p0s (numpy array): |S| array of state starting probabilities
        t_mat (numpy array): |S|x|A|x|S| array of transition probabilities
        terminal_state_mask (numpy array): |S| array indicating terminal states
        rs (numpy array): |S| array of state reward weights
        max_path_length (int): Maximum path length to consider
        
    Returns:
        (numpy array): |S| array of un-discounted state marginals
    """

    # Prepare state and action partition arrays (Step 1)
    # NB: This step is different to the '08 version
    Z_s = np.zeros(t_mat.shape[0])
    Z_a = np.zeros(t_mat.shape[1])
    Z_s[terminal_state_mask] = 1

    # Compute local partition values (Step 2)
    for t in range(max_path_length):
        # Sweep actions
        Z_a[:] = 0
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    Z_a[a] += t_mat[s1, a, s2] * np.exp(rs[s1]) * Z_s[s2]

        # Sweep states
        # NB: This step is different to the '08 version
        Z_s[:] = 0
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    Z_s[s1] += Z_a[a] + terminal_state_mask[s1]

    # Compute local action probabilities (Step 3)
    prob_sa = np.zeros(t_mat.shape[0 : 1 + 1])
    for s1 in range(t_mat.shape[0]):
        for a in range(t_mat.shape[1]):
            if Z_s[s1] == 0.0:
                # This state was never reached during the backward pass
                # Default to a uniform distribution over actions
                prob_sa[s1, :] = 1.0 / t_mat.shape[1]
            else:
                prob_sa[s1, a] = Z_a[a] / Z_s[s1]

    # Prepare state marginal array (Step 4)
    dst = np.zeros((t_mat.shape[0], max_path_length + 1))
    for t in range(max_path_length + 1):
        dst[:, t] = p0s

    # Compute state occurrences at each time (Step 5)
    for t in range(max_path_length):
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    # NB: This step is different to the '08 version
                    dst[s2, t + 1] += dst[s1, t] * prob_sa[s1, a] * t_mat[s1, a, s2]

    # Sum frequencies to get state marginals (Step 6)
    ps = np.sum(dst, axis=1)
    return ps


@jit(nopython=True)
def nb_state_marginals_10(p0s, t_mat, terminal_state_mask, rs, max_path_length):
    """Compute state marginals using Ziebart's 2010 algorithm
    
    This algorithm is designed to compute state marginals for an un-discounted MDP with
    state-feature reward basis functions.
    
    Args:
        p0s (numpy array): |S| array of state starting probabilities
        t_mat (numpy array): |S|x|A|x|S| array of transition probabilities
        terminal_state_mask (numpy array): |S| array indicating terminal states
        rs (numpy array): |S| array of state reward weights
        max_path_length (int): Maximum path length to consider
        
    Returns:
        (numpy array): |S| array of un-discounted state marginals
    """

    # Prepare state and action partition arrays (Step 1)
    # NB: This step is different to the '08 version
    Z_s = np.zeros(t_mat.shape[0])
    Z_a = np.zeros(t_mat.shape[1])
    Z_s[terminal_state_mask] = 1

    # Compute local partition values (Step 2)
    for t in range(max_path_length):
        # Sweep actions
        Z_a[:] = 0
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    Z_a[a] += t_mat[s1, a, s2] * np.exp(rs[s1]) * Z_s[s2]

        # Sweep states
        # NB: This step is different to the '08 version
        Z_s[:] = 0
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    Z_s[s1] += Z_a[a] + terminal_state_mask[s1]

    # Compute local action probabilities (Step 3)
    prob_sa = np.zeros(t_mat.shape[0 : 1 + 1])
    for s1 in range(t_mat.shape[0]):
        for a in range(t_mat.shape[1]):
            if Z_s[s1] == 0.0:
                # This state was never reached during the backward pass
                # Default to a uniform distribution over actions
                prob_sa[s1, :] = 1.0 / t_mat.shape[1]
            else:
                prob_sa[s1, a] = Z_a[a] / Z_s[s1]

    # Prepare state marginal array (Step 4)
    dst = np.zeros((t_mat.shape[0], max_path_length + 1))
    for t in range(max_path_length + 1):
        dst[:, t] = p0s

    # Compute state occurrences at each time (Step 5)
    for t in range(max_path_length):
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    # NB: This step is different to the '08 version
                    dst[s2, t + 1] += dst[s1, t] * prob_sa[s1, a] * t_mat[s1, a, s2]

    # Sum frequencies to get state marginals (Step 6)
    ps = np.sum(dst, axis=1)
    return ps


def nll_s(theta_s, env, max_path_length, version, phibar_s, rescale_grad, verbose):
    nll_s._call_count += 1
    if verbose:
        print("Obj#{}".format(nll_s._call_count))
        print(theta_s)
    env._state_rewards = theta_s
    with np.errstate(over="raise"):

        # We use our algorithm to find the true value of the NLL
        alpha_log = backward_pass_log(
            env.p0s, max_path_length, env.t_mat, env.gamma, rs=theta_s
        )
        Z_log = partition_log(max_path_length, alpha_log, with_dummy_state=False)
        nll = Z_log - theta_s @ phibar_s

        # We use Ziebart's algorithm(s) to find the gradient
        if version == "08":
            # Compute state marginals
            ps = state_marginals_08(
                env.p0s, env.t_mat, env.state_rewards, max_path_length
            )
        elif version == "10":
            ps = state_marginals_10(
                env.p0s,
                env.t_mat,
                env.terminal_state_mask,
                env.state_rewards,
                max_path_length,
            )
        else:
            raise ValueError(f"Version must be one of '08' or '10', was {version}")

        grad = ps - phibar_s

        if rescale_grad:
            if verbose:
                print(f"Re-scaling gradient by a factor of 1/{np.linalg.norm(grad)}")
            grad /= np.linalg.norm(grad)

        if verbose:
            print("Grad = ")
            print(grad)

    return nll, grad


# Static objective function call count
nll_s._call_count = 0


def zb_maxent_irl(
    rollouts,
    env,
    rbound=(-1.0, 1.0),
    version="10",
    step_size=1e-4,
    tol=1e-6,
    max_iterations=None,
    verbose=False,
):
    """Maximum Entropy IRL
    
    Args:
        rollouts (list): List of [(s, a), (s, a), ..., (s, None)] trajectories
        env (.envs.explicit.IExplicitEnv) Environment to solve
        
        rbound (float): Minimum and maximum reward weight values
        version (str): Which version of Ziebart's algorithm to use.
            Must be one of '08' or '10'
        rescale_grad (bool): If true, re-scale the gradient by it's L2 norm. This can
            help prevent error message 'ABNORMAL_TERMINATION_IN_LNSRCH' from L-BFGS-B
            for some problems.
        verbose (bool): Extra logging
    
    Returns:
        (numpy array): State reward weights
    """

    assert version in (
        "08",
        "10",
    ), f"Version must be one of '08' or '10', was {version}"
    assert env.gamma == 1.0, "Ziebart's algorithms only support un-discounted MDPs"

    # Copy the environment so we don't modify it
    env = copy.deepcopy(env)

    num_states = len(env.states)
    num_actions = len(env.actions)

    max_path_length = max([len(r) for r in rollouts])
    if verbose:
        print("Max path length: {}".format(max_path_length))

    # Find discounted feature expectations
    phibar_s, _, _ = empirical_feature_expectations(env, rollouts)

    # Begin gradient descent loop
    if verbose:
        print(f"Optimizing state rewards with '{version} algorithm")
    theta_s = np.zeros(num_states)
    for gd_iteration in it.count():
        env._state_rewards = theta_s
        if verbose:
            print("Obj#{}".format(gd_iteration))
            print(theta_s)

        if version == "08":
            # Compute state marginals
            ps = state_marginals_08(
                env.p0s, env.t_mat, env.children, env.state_rewards, max_path_length
            )
        elif version == "10":
            ps = state_marginals_10(
                env.p0s,
                env.t_mat,
                env.terminal_state_mask,
                env.children,
                env.state_rewards,
                max_path_length,
            )
        else:
            raise ValueError(f"Version must be one of '08' or '10', was {version}")

        # Compute gradient descent gradient
        gd_grad = ps - phibar_s
        if verbose:
            print(f"Gradient = {gd_grad}")

        # Step along gradient
        theta_s_new = theta_s + step_size * gd_grad

        # Clamp reward weights to bounds
        theta_s_new_clipped = np.clip(theta_s_new, *rbound)
        clamped_weights = int(np.sum(np.abs(theta_s_new - theta_s_new_clipped) > 0))
        if clamped_weights > 0 and verbose:
            print(f"Clipping {clamped_weights} weight values to range {rbound}")
        theta_s_new = theta_s_new_clipped

        # Measure weight change
        delta = np.max(np.abs(theta_s_new - theta_s))
        theta_s = theta_s_new

        if verbose:
            print(f"Weight delta: {delta}")

        if delta < tol:
            if verbose:
                print("Reached weight convergence - stopping")
            break

        if max_iterations is not None and gd_iteration == max_iterations - 1:
            if verbose:
                print("Reached max iterations - stopping")
            break

    return theta_s
