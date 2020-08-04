"""Implements Maximum Entropy IRL from my thesis"""

import copy
import numpy as np
from numba import jit
from scipy.optimize import minimize

from unimodal_irl.utils import empirical_feature_expectations


# Placeholder for 'negative infinity' which doesn't cause NaN in log-space operations
_NINF = np.finfo(np.float64).min


def backward_pass_log(p0s, L, t_mat, gamma=1.0, rs=None, rsa=None, rsas=None):
    """Compute backward message passing variable in log-space
    
    Args:
        p0s (numpy array): Starting state probabilities
        L (int): Maximum path length
        t_mat (numpy array): |S|x|A|x|S| transition matrix
        
        gamma (float): Discount factor
        rs (numpy array): |S| array of linear state reward weights
        rsa (numpy array): |S|x|A| array of linear state-action reward weights
        rsas (numpy array): Linear state-action-state reward weights
    
    Returns:
        (numpy array): |S|xL array of backward message values in log space
    """

    if rs is None:
        rs = np.zeros(t_mat.shape[0])
    if rsa is None:
        rsa = np.zeros(t_mat.shape[0:2])
    if rsas is None:
        rsas = np.zeros(t_mat.shape[0:3])

    alpha = np.zeros((t_mat.shape[0], L))
    with np.errstate(divide="ignore"):
        alpha[:, 0] = np.log(p0s) + rs
    for t in range(L - 1):
        for s2 in range(t_mat.shape[2]):
            # Find maximum value among all parents of s2
            m_t = _NINF
            for s1 in range(t_mat.shape[0]):
                for a in range(t_mat.shape[1]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    m_t = max(
                        m_t,
                        (
                            alpha[s1, t]
                            + np.log(t_mat[s1, a, s2])
                            + gamma ** ((t + 1) - 1) * (rsa[s1, a] + rsas[s1, a, s2])
                        ),
                    )
            m_t += (gamma ** (t + 1)) * rs[s2]

            # Compute next column of alpha in log-space
            for s1 in range(t_mat.shape[0]):
                for a in range(t_mat.shape[1]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    alpha[s2, t + 1] += t_mat[s1, a, s2] * np.exp(
                        alpha[s1, t]
                        + gamma ** ((t + 1) - 1) * (rsa[s1, a] + rsas[s1, a, s2])
                        + (gamma ** (t + 1)) * rs[s2]
                        - m_t
                    )
            with np.errstate(divide="ignore"):
                alpha[s2, t + 1] = m_t + np.log(alpha[s2, t + 1])

    return alpha


@jit(nopython=True)
def nb_backward_pass_log(p0s, L, t_mat, gamma=1.0, rs=None, rsa=None, rsas=None):
    """Compute backward message passing variable in log-space
    
    Args:
        p0s (numpy array): Starting state probabilities
        L (int): Maximum path length
        t_mat (numpy array): |S|x|A|x|S| transition matrix
        
        gamma (float): Discount factor
        rs (numpy array): |S| array of linear state reward weights
        rsa (numpy array): |S|x|A| array of linear state-action reward weights
        rsas (numpy array): Linear state-action-state reward weights
    
    Returns:
        (numpy array): |S|xL array of backward message values in log space
    """

    if rs is None:
        rs = np.zeros(t_mat.shape[0])
    if rsa is None:
        rsa = np.zeros(t_mat.shape[0:2])
    if rsas is None:
        rsas = np.zeros(t_mat.shape[0:3])

    alpha = np.zeros((t_mat.shape[0], L))
    alpha[:, 0] = np.log(p0s) + rs
    for t in range(L - 1):
        for s2 in range(t_mat.shape[2]):
            # Find maximum value among all parents of s2
            m_t = _NINF
            for s1 in range(t_mat.shape[0]):
                for a in range(t_mat.shape[1]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    m_t = max(
                        m_t,
                        (
                            alpha[s1, t]
                            + np.log(t_mat[s1, a, s2])
                            + gamma ** ((t + 1) - 1) * (rsa[s1, a] + rsas[s1, a, s2])
                        ),
                    )
            m_t += (gamma ** (t + 1)) * rs[s2]

            # Compute next column of alpha in log-space
            for s1 in range(t_mat.shape[0]):
                for a in range(t_mat.shape[1]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    alpha[s2, t + 1] += t_mat[s1, a, s2] * np.exp(
                        alpha[s1, t]
                        + gamma ** ((t + 1) - 1) * (rsa[s1, a] + rsas[s1, a, s2])
                        + (gamma ** (t + 1)) * rs[s2]
                        - m_t
                    )
            alpha[s2, t + 1] = m_t + np.log(alpha[s2, t + 1])

    return alpha


def forward_pass_log(L, t_mat, gamma=1.0, rs=None, rsa=None, rsas=None):
    """Compute forward message passing variable in log space
    
    Args:
        L (int): Maximum path length
        t_mat (numpy array): |S|x|A|x|S| transition matrix
        children (dict): Dictionary mapping states to (a, s') child tuples
        
        gamma (float): Discount factor
        rs (numpy array): Linear state reward weights
        rsa (numpy array): Linear state-action reward weights
        rsas (numpy array): Linear state-action-state reward weights
    
    Returns:
        (numpy array): |S| x L array of forward message values in log space
    """

    if rs is None:
        rs = np.zeros(t_mat.shape[0])
    if rsa is None:
        rsa = np.zeros(t_mat.shape[0:2])
    if rsas is None:
        rsas = np.zeros(t_mat.shape[0:3])

    beta = np.zeros((t_mat.shape[0], L))
    beta[:, 0] = gamma ** (L - 1) * rs
    for t in range(L - 1):
        for s1 in range(t_mat.shape[0]):
            # Find maximum value among children of s1
            m_t = _NINF
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    m_t = max(
                        m_t,
                        (
                            np.log(t_mat[s1, a, s2])
                            + gamma ** (L - (t + 1) - 1)
                            * (rsa[s1, a] + rsas[s1, a, s2])
                            + beta[s2, t]
                        ),
                    )
            m_t += gamma ** (L - (t + 1) - 1) * rs[s1]

            # Compute next column of beta in log-space
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    beta[s1, t + 1] += t_mat[s1, a, s2] * np.exp(
                        gamma ** (L - (t + 1) - 1)
                        * (rs[s1] + rsa[s1, a] + rsas[s1, a, s2])
                        + beta[s2, t]
                        - m_t
                    )
            beta[s1, t + 1] = m_t + np.log(beta[s1, t + 1])

    return beta


@jit(nopython=True)
def nb_forward_pass_log(L, t_mat, gamma=1.0, rs=None, rsa=None, rsas=None):
    """Compute forward message passing variable in log space
    
    Args:
        L (int): Maximum path length
        t_mat (numpy array): |S|x|A|x|S| transition matrix
        children (dict): Dictionary mapping states to (a, s') child tuples
        
        gamma (float): Discount factor
        rs (numpy array): Linear state reward weights
        rsa (numpy array): Linear state-action reward weights
        rsas (numpy array): Linear state-action-state reward weights
    
    Returns:
        (numpy array): |S| x L array of forward message values in log space
    """

    if rs is None:
        rs = np.zeros(t_mat.shape[0])
    if rsa is None:
        rsa = np.zeros(t_mat.shape[0:2])
    if rsas is None:
        rsas = np.zeros(t_mat.shape[0:3])

    beta = np.zeros((t_mat.shape[0], L))
    beta[:, 0] = gamma ** (L - 1) * rs
    for t in range(L - 1):
        for s1 in range(t_mat.shape[0]):
            # Find maximum value among children of s1
            m_t = _NINF
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    m_t = max(
                        m_t,
                        (
                            np.log(t_mat[s1, a, s2])
                            + gamma ** (L - (t + 1) - 1)
                            * (rsa[s1, a] + rsas[s1, a, s2])
                            + beta[s2, t]
                        ),
                    )
            m_t += gamma ** (L - (t + 1) - 1) * rs[s1]

            # Compute next column of beta in log-space
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    beta[s1, t + 1] += t_mat[s1, a, s2] * np.exp(
                        gamma ** (L - (t + 1) - 1)
                        * (rs[s1] + rsa[s1, a] + rsas[s1, a, s2])
                        + beta[s2, t]
                        - m_t
                    )
            beta[s1, t + 1] = m_t + np.log(beta[s1, t + 1])

    return beta


def partition_log(L, alpha_log, with_dummy_state=True):
    """Compute the partition function
    
    Args:
        L (int): Maximum path length
        alpha_log (numpy array): |S|xL backward message variable in log space
        
        with_dummy_state (bool): If true, the final row of the alpha matrix corresponds
            to a dummy state which is used for MDP padding
        
    Returns:
        (float): Partition function value
    """

    # If the dummy state is included, don't include it in the partition
    if with_dummy_state:
        alpha_log = alpha_log[0:-1, :]

    # Find maximum value
    m = np.max(alpha_log[:, 0:L])

    # Compute partition in log space
    return m + np.log(np.sum(np.exp(alpha_log[:, 0:L] - m)))


def marginals_log(
    L, t_mat, alpha_log, beta_log, Z_theta_log, gamma=1.0, rsa=None, rsas=None
):
    """Compute marginal terms
    
    Args:
        L (int): Maximum path length
        t_mat (numpy array): |S|x|A|x|S| transition matrix
        alpha_log (numpy array): |S|xL array of backward message values in log space
        beta_log (numpy array): |S|xL array of forward message values in log space
        Z_theta_log (float): Partition value in log space
        
        gamma (float): Discount factor
        rsa (numpy array): |S|x|A| array of linear state-action reward weights
        rsas (numpy array): |S|x|A|x|S| array of linear state-action-state reward
            weights
    
    Returns:
        (numpy array): |S|xL array of state marginals in log space
        (numpy array): |S|x|A|xL array of state-action marginals in log space
        (numpy array): |S|x|A|x|S|xL array of state-action-state marginals in log space
    """

    if rsa is None:
        rsa = np.zeros((t_mat.shape[0], t_mat.shape[1]))
    if rsas is None:
        rsas = np.zeros((t_mat.shape[0], t_mat.shape[1], t_mat.shape[2]))

    pts = np.zeros((t_mat.shape[0], L))
    ptsa = np.zeros((t_mat.shape[0], t_mat.shape[1], L - 1))
    ptsas = np.zeros((t_mat.shape[0], t_mat.shape[1], t_mat.shape[2], L - 1))

    for t in range(L - 1):

        for s1 in range(t_mat.shape[0]):

            # if np.isneginf(alpha_log[s1, t]):
            if np.exp(alpha_log[s1, t]) == 0:
                # Catch edge case where the backward message value is zero to prevent
                # floating point error
                pts[s1, t] = -np.inf
                ptsa[s1, :, t] = -np.inf
                ptsas[s1, :, :, t] = -np.inf
            else:
                # Compute max value
                m_t = _NINF
                for a in range(t_mat.shape[1]):
                    for s2 in range(t_mat.shape[2]):
                        if t_mat[s1, a, s2] != 0:
                            m_t = max(
                                m_t,
                                (
                                    np.log(t_mat[s1, a, s2])
                                    + gamma ** ((t + 1) - 1)
                                    * (rsa[s1, a] + rsas[s1, a, s2])
                                    + beta_log[s2, L - (t + 1) - 1]
                                ),
                            )
                m_t += alpha_log[s1, t] - Z_theta_log

                # Compute state marginals in log space
                for a in range(t_mat.shape[1]):
                    for s2 in range(t_mat.shape[2]):
                        contrib = t_mat[s1, a, s2] * np.exp(
                            alpha_log[s1, t]
                            + gamma ** ((t + 1) - 1) * (rsa[s1, a] + rsas[s1, a, s2])
                            + beta_log[s2, L - (t + 1) - 1]
                            - Z_theta_log
                            - m_t
                        )
                        pts[s1, t] += contrib
                        ptsa[s1, a, t] += contrib
                        if contrib == 0:
                            ptsas[s1, a, s2, t] = -np.inf
                        else:
                            ptsas[s1, a, s2, t] = m_t + np.log(contrib)
                    if ptsa[s1, a, t] == 0:
                        ptsa[s1, a, t] = -np.inf
                    else:
                        ptsa[s1, a, t] = m_t + np.log(ptsa[s1, a, t])
                if pts[s1, t] == 0:
                    pts[s1, t] = -np.inf
                else:
                    pts[s1, t] = m_t + np.log(pts[s1, t])

    # Compute final column of pts
    pts[:, L - 1] = alpha_log[:, L - 1] - Z_theta_log

    return pts, ptsa, ptsas


@jit(nopython=True)
def nb_marginals_log(
    L, t_mat, alpha_log, beta_log, Z_theta_log, gamma=1.0, rsa=None, rsas=None
):
    """Compute marginal terms
    
    Args:
        L (int): Maximum path length
        t_mat (numpy array): |S|x|A|x|S| transition matrix
        alpha_log (numpy array): |S|xL array of backward message values in log space
        beta_log (numpy array): |S|xL array of forward message values in log space
        Z_theta_log (float): Partition value in log space
        
        gamma (float): Discount factor
        rsa (numpy array): |S|x|A| array of linear state-action reward weights
        rsas (numpy array): |S|x|A|x|S| array of linear state-action-state reward
            weights
    
    Returns:
        (numpy array): |S|xL array of state marginals in log space
        (numpy array): |S|x|A|xL array of state-action marginals in log space
        (numpy array): |S|x|A|x|S|xL array of state-action-state marginals in log space
    """

    if rsa is None:
        rsa = np.zeros((t_mat.shape[0], t_mat.shape[1]))
    if rsas is None:
        rsas = np.zeros((t_mat.shape[0], t_mat.shape[1], t_mat.shape[2]))

    pts = np.zeros((t_mat.shape[0], L))
    ptsa = np.zeros((t_mat.shape[0], t_mat.shape[1], L - 1))
    ptsas = np.zeros((t_mat.shape[0], t_mat.shape[1], t_mat.shape[2], L - 1))

    for t in range(L - 1):

        for s1 in range(t_mat.shape[0]):

            # if np.isneginf(alpha_log[s1, t]):
            if np.exp(alpha_log[s1, t]) == 0:
                # Catch edge case where the backward message value is zero to prevent
                # floating point error
                pts[s1, t] = -np.inf
                ptsa[s1, :, t] = -np.inf
                ptsas[s1, :, :, t] = -np.inf
            else:
                # Compute max value
                m_t = _NINF
                for a in range(t_mat.shape[1]):
                    for s2 in range(t_mat.shape[2]):
                        if t_mat[s1, a, s2] != 0:
                            m_t = max(
                                m_t,
                                (
                                    np.log(t_mat[s1, a, s2])
                                    + gamma ** ((t + 1) - 1)
                                    * (rsa[s1, a] + rsas[s1, a, s2])
                                    + beta_log[s2, L - (t + 1) - 1]
                                ),
                            )
                m_t += alpha_log[s1, t] - Z_theta_log

                # Compute state marginals in log space
                for a in range(t_mat.shape[1]):
                    for s2 in range(t_mat.shape[2]):
                        contrib = t_mat[s1, a, s2] * np.exp(
                            alpha_log[s1, t]
                            + gamma ** ((t + 1) - 1) * (rsa[s1, a] + rsas[s1, a, s2])
                            + beta_log[s2, L - (t + 1) - 1]
                            - Z_theta_log
                            - m_t
                        )
                        pts[s1, t] += contrib
                        ptsa[s1, a, t] += contrib
                        if contrib == 0:
                            ptsas[s1, a, s2, t] = -np.inf
                        else:
                            ptsas[s1, a, s2, t] = m_t + np.log(contrib)
                    if ptsa[s1, a, t] == 0:
                        ptsa[s1, a, t] = -np.inf
                    else:
                        ptsa[s1, a, t] = m_t + np.log(ptsa[s1, a, t])
                if pts[s1, t] == 0:
                    pts[s1, t] = -np.inf
                else:
                    pts[s1, t] = m_t + np.log(pts[s1, t])

    # Compute final column of pts
    pts[:, L - 1] = alpha_log[:, L - 1] - Z_theta_log

    return pts, ptsa, ptsas


def env_solve(env, L, with_dummy_state=True):
    """Convenience method to solve an environment for marginals
    
    Args:
        env (.envs.explicit.IExplicitEnv) Environment to solve
        L (int): Max path length
        with_dummy_state (bool): Indicates if the environment has been padded with a
            dummy state and action, or not
    
    Returns:
        (numpy array): |S|xL array of state marginals in log space
        (numpy array): |S|x|A|xL array of state-action marginals in log space
        (numpy array): |S|x|A|x|S|xL array of state-action-state marginals in log space
        (float): Partition value in log space
    """

    # Numba has trouble typing these arguments to the forward/backward functions below.
    # We manually convert them here to avoid typing issues at JIT compile time
    rs = env.state_rewards
    if rs is None:
        rs = np.zeros(env.t_mat.shape[0], dtype=np.float)

    rsa = env.state_action_rewards
    if rsa is None:
        rsa = np.zeros(env.t_mat.shape[0:2], dtype=np.float)

    rsas = env.state_action_state_rewards
    if rsas is None:
        rsas = np.zeros(env.t_mat.shape[0:3], dtype=np.float)

    alpha_log = nb_backward_pass_log(
        env.p0s, L, env.t_mat, gamma=env.gamma, rs=rs, rsa=rsa, rsas=rsas,
    )
    beta_log = nb_forward_pass_log(
        L, env.t_mat, gamma=env.gamma, rs=rs, rsa=rsa, rsas=rsas,
    )
    Z_theta_log = partition_log(L, alpha_log, with_dummy_state=with_dummy_state)
    pts_log, ptsa_log, ptsas_log = nb_marginals_log(
        L,
        env.t_mat,
        alpha_log,
        beta_log,
        Z_theta_log,
        gamma=env.gamma,
        rsa=rsa,
        rsas=rsas,
    )
    return pts_log, ptsa_log, ptsas_log, Z_theta_log


def nll_s(
    theta_s, env, max_path_length, with_dummy_state, phibar_s, rescale_grad, verbose
):
    nll_s._call_count += 1
    if verbose:
        print("Obj#{}".format(nll_s._call_count))
        print(theta_s)
    env._state_rewards = theta_s
    with np.errstate(over="raise"):
        pts_log, _, _, Z_log = env_solve(
            env, max_path_length, with_dummy_state=with_dummy_state
        )
        nll = Z_log - theta_s @ phibar_s
        grad = np.sum(np.exp(pts_log), axis=1) - phibar_s
        if rescale_grad:
            if verbose:
                print(f"Re-scaling gradient by a factor of 1/{np.linalg.norm(grad)}")
            grad /= np.linalg.norm(grad)
    return nll, grad


# Static objective function call count
nll_s._call_count = 0


def nll_sa(
    theta_sa, env, max_path_length, with_dummy_state, phibar_sa, rescale_grad, verbose
):
    nll_sa._call_count += 1
    if verbose:
        print("Obj#{}".format(nll_sa._call_count))
        print(theta_sa)
    env._state_action_rewards = theta_sa.reshape((len(env.states), len(env.actions)))
    with np.errstate(over="raise"):
        _, ptsa_log, _, Z_log = env_solve(
            env, max_path_length, with_dummy_state=with_dummy_state
        )
        nll = Z_log - theta_sa @ phibar_sa.flatten()
        grad = (np.sum(np.exp(ptsa_log), axis=2) - phibar_sa).flatten()
        if rescale_grad:
            if verbose:
                print(f"Re-scaling gradient by a factor of 1/{np.linalg.norm(grad)}")
            grad /= np.linalg.norm(grad)
    return nll, grad


# Static objective function call count
nll_sa._call_count = 0


def nll_sas(
    theta_sas, env, max_path_length, with_dummy_state, phibar_sas, rescale_grad, verbose
):
    nll_sas._call_count += 1
    if verbose:
        print("Obj#{}".format(nll_sas._call_count))
        print(theta_sas)
    env._state_action_state_rewards = theta_sas.reshape(
        (len(env.states), len(env.actions), len(env.states))
    )
    with np.errstate(over="raise"):
        _, _, ptsas_log, Z_log = env_solve(
            env, max_path_length, with_dummy_state=with_dummy_state
        )
        nll = Z_log - theta_sas @ phibar_sas.flatten()
        grad = (np.sum(np.exp(ptsas_log), axis=3) - phibar_sas).flatten()
        if rescale_grad:
            if verbose:
                print(f"Re-scaling gradient by a factor of 1/{np.linalg.norm(grad)}")
            grad /= np.linalg.norm(grad)
    return nll, grad


# Static objective function call count
nll_sas._call_count = 0


def sw_maxent_irl(
    rollouts,
    env,
    rs=False,
    rsa=False,
    rsas=False,
    rbound=(-1.0, 1.0),
    with_dummy_state=False,
    rescale_grad=False,
    verbose=False,
):
    """Maximum Entropy IRL
    
    Args:
        rollouts (list): List of [(s, a), (s, a), ..., (s, None)] trajectories
        env (.envs.explicit.IExplicitEnv) Environment to solve
        
        rs (bool): Optimize for state rewards?
        rsa (bool): Optimize for state-action rewards?
        rsas (bool): Optimize for state-action-state rewards?
        rbound (float): Minimum and maximum reward weight values
        with_dummy_state (bool): Indicates if the MDP has been padded to include a dummy
            state
        rescale_grad (bool): If true, re-scale the gradient by it's L2 norm. This can
            help prevent error message 'ABNORMAL_TERMINATION_IN_LNSRCH' from L-BFGS-B
            for some problems.
        verbose (bool): Extra logging
    
    Returns:
        (numpy array): State reward weights
        (numpy array): State-action reward weights
        (numpy array): State-action-state reward weights
    """

    assert rs or rsa or rsas, "Must request at least one of rs, rsa or rsas"

    # Copy the environment so we don't modify it
    env = copy.deepcopy(env)

    num_states = len(env.states)
    num_actions = len(env.actions)

    # Find max path length
    if len(rollouts) == 1:
        max_path_length = min_path_length = len(rollouts[0])
    else:
        max_path_length = max(*[len(r) for r in rollouts])
        min_path_length = min(*[len(r) for r in rollouts])
    if verbose:
        print("Max path length: {}".format(max_path_length))

    # Detect if the env should have been padded
    assert (
        min_path_length == max_path_length
    ), "Paths are of unequal lengths - please ensure the environment is padded before continuing"

    # Find discounted feature expectations
    phibar_s, phibar_sa, phibar_sas = empirical_feature_expectations(env, rollouts)

    # Reset objective function call counts
    nll_s._call_count = 0
    nll_sa._call_count = 0
    nll_sas._call_count = 0
    theta_s = None
    if rs:
        if verbose:
            print("Optimizing state rewards")
        res = minimize(
            nll_s,
            np.zeros(num_states),
            args=(
                env,
                max_path_length,
                with_dummy_state,
                phibar_s,
                rescale_grad,
                verbose,
            ),
            # method="L-BFGS-B",
            method="TNC",
            jac=True,
            bounds=tuple(rbound for _ in range(num_states)),
        )
        theta_s = res.x
        if verbose:
            print(res)
            print(
                "Completed optimization after {} iterations, NLL = {}".format(
                    res.nit, res.fun
                )
            )
            print("theta_s = {}".format(theta_s))

    theta_sa = None
    if rsa:
        if verbose:
            print("Optimizing state-action rewards")
        res = minimize(
            nll_sa,
            np.zeros(num_states * num_actions),
            args=(
                env,
                max_path_length,
                with_dummy_state,
                phibar_sa,
                rescale_grad,
                verbose,
            ),
            # method="L-BFGS-B",
            method="TNC",
            jac=True,
            bounds=tuple(rbound for _ in range(num_states * num_actions)),
        )
        theta_sa = res.x.reshape((num_states, num_actions))
        if verbose:
            print(res)
            print(
                "Completed optimization after {} iterations, NLL = {}".format(
                    res.nit, res.fun
                )
            )
            print("theta_sa = {}".format(theta_sa))

    theta_sas = None
    if rsas:
        if verbose:
            print("Optimizing state-action-state rewards")
        res = minimize(
            nll_sas,
            np.zeros(num_states * num_actions * num_states),
            args=(
                env,
                max_path_length,
                with_dummy_state,
                phibar_sas,
                rescale_grad,
                verbose,
            ),
            # method="L-BFGS-B",
            method="TNC",
            jac=True,
            bounds=tuple(rbound for _ in range(num_states * num_actions * num_states)),
        )
        theta_sas = res.x.reshape((num_states, num_actions, num_states))
        if verbose:
            print(res)
            print(
                "Completed optimization after {} iterations, NLL = {}".format(
                    res.nit, res.fun
                )
            )
            print("theta_sas = {}".format(theta_sas))

    return theta_s, theta_sa, theta_sas


def main():
    """Main function"""
    pass


if __name__ == "__main__":
    main()
