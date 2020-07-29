"""Implements Maximum Entropy IRL from my thesis"""

import copy
import warnings
import numpy as np
from scipy.optimize import minimize

from unimodal_irl.envs.utils import pad_terminal_mdp

# Placeholder for 'negative infinity' which doesn't cause NaN in log-space operations
_NINF = np.finfo(np.float64).min


def backward_pass(p0s, L, t_mat, parents, gamma=1.0, rs=None, rsa=None, rsas=None):
    """Compute backward message passing variable
    
    Args:
        p0s (numpy array): Starting state probabilities
        L (int): Maximum path length
        t_mat (numpy array): |S|x|A|x|S| transition matrix
        parents (dict): Dictionary mapping states to (s, a) parent tuples
        
        gamma (float): Discount factor
        rs (numpy array): |S| array of linear state reward weights
        rsa (numpy array): |S|x|A| array of linear state-action reward weights
        rsas (numpy array): |S|x|A|x|S| array of linear state-action-state reward
            weights
    
    Returns:
        (numpy array): |S|xL array of backward message values
    """

    if rs is None:
        rs = np.zeros(t_mat.shape[0])
    if rsa is None:
        rsa = np.zeros(t_mat.shape[0:2])
    if rsas is None:
        rsas = np.zeros(t_mat.shape[0:3])

    alpha = np.zeros((t_mat.shape[0], L))
    alpha[:, 0] = p0s * np.exp(rs)
    for t in range(L - 1):
        for s2 in range(t_mat.shape[0]):
            for s1, a in parents[s2]:
                alpha[s2, t + 1] += (
                    alpha[s1, t]
                    * t_mat[s1, a, s2]
                    * np.exp(
                        gamma ** ((t + 1) - 1) * (rsa[s1, a] + rsas[s1, a, s2])
                        + (gamma ** (t + 1)) * rs[s2]
                    )
                )
    return alpha


def backward_pass_log(p0s, L, t_mat, parents, gamma=1.0, rs=None, rsa=None, rsas=None):
    """Compute backward message passing variable in log-space
    
    Args:
        p0s (numpy array): Starting state probabilities
        L (int): Maximum path length
        t_mat (numpy array): |S|x|A|x|S| transition matrix
        parents (dict): Dictionary mapping states to (s, a) parent tuples
        
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
        for s2 in range(t_mat.shape[0]):
            # Find maximum value among all parents of s2
            m_t = _NINF
            for s1, a in parents[s2]:
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
            for s1, a in parents[s2]:
                alpha[s2, t + 1] += t_mat[s1, a, s2] * np.exp(
                    alpha[s1, t]
                    + gamma ** ((t + 1) - 1) * (rsa[s1, a] + rsas[s1, a, s2])
                    + (gamma ** (t + 1)) * rs[s2]
                    - m_t
                )
            with np.errstate(divide="ignore"):
                alpha[s2, t + 1] = m_t + np.log(alpha[s2, t + 1])

    return alpha


def env_backward(env, L):
    """Convenience method for backward message passing
    
    Args:
        env (.envs.explicit_env.IExplicitEnv) Environment to solve
        L (int): Max path length
    
    Returns:
        (numpy array): |S|xL array of backward message values in log space
    """
    return backward_pass_log(
        env.p0s,
        L,
        env.t_mat,
        env.parents,
        gamma=env.gamma,
        rs=env.state_rewards,
        rsa=env.state_action_rewards,
        rsas=env.state_action_state_rewards,
    )


def forward_pass(L, t_mat, children, gamma=1.0, rs=None, rsa=None, rsas=None):
    """Compute forward message passing variable
    
    Args:
        L (int): Maximum path length
        t_mat (numpy array): |S|x|A|x|S| transition matrix
        children (dict): Dictionary mapping states to (a, s') child tuples
        
        gamma (float): Discount factor
        rs (numpy array): |S| array of linear state reward weights
        rsa (numpy array): |S|x|A| array of linear state-action reward weights
        rsas (numpy array): Linear state-action-state reward weights
    
    Returns:
        (numpy array): |S|xL array of forward message values
    """

    if rs is None:
        rs = np.zeros(t_mat.shape[0])
    if rsa is None:
        rsa = np.zeros(t_mat.shape[0:2])
    if rsas is None:
        rsas = np.zeros(t_mat.shape[0:3])

    beta = np.zeros((t_mat.shape[0], L))
    beta[:, 0] = np.exp(gamma ** (L - 1) * rs)
    for t in range(L - 1):
        for s1 in range(t_mat.shape[0]):
            for a, s2 in children[s1]:
                beta[s1, t + 1] += (
                    t_mat[s1, a, s2]
                    * np.exp(
                        gamma ** (L - (t + 1) - 1)
                        * (rs[s1] + rsa[s1, a] + rsas[s1, a, s2])
                    )
                    * beta[s2, t]
                )
    return beta


def forward_pass_log(L, t_mat, children, gamma=1.0, rs=None, rsa=None, rsas=None):
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
            for a, s2 in children[s1]:
                m_t = max(
                    m_t,
                    (
                        np.log(t_mat[s1, a, s2])
                        + gamma ** (L - (t + 1) - 1) * (rsa[s1, a] + rsas[s1, a, s2])
                        + beta[s2, t]
                    ),
                )
            m_t += gamma ** (L - (t + 1) - 1) * rs[s1]

            # Compute next column of beta in log-space
            for a, s2 in children[s1]:
                beta[s1, t + 1] += t_mat[s1, a, s2] * np.exp(
                    gamma ** (L - (t + 1) - 1) * (rs[s1] + rsa[s1, a] + rsas[s1, a, s2])
                    + beta[s2, t]
                    - m_t
                )
            beta[s1, t + 1] = m_t + np.log(beta[s1, t + 1])

    return beta


def env_forward(env, L):
    """Convenience method for forward message passing
    
    Args:
        env (.envs.explicit_env.IExplicitEnv) Environment to solve
        L (int): Max path length
    
    Returns:
        (numpy array): |S|xL array of forward message values in log space
    """
    return forward_pass_log(
        L,
        env.t_mat,
        env.children,
        gamma=env.gamma,
        rs=env.state_rewards,
        rsa=env.state_action_rewards,
        rsas=env.state_action_state_rewards,
    )


def partition(L, alpha, with_dummy_state=True):
    """Compute the partition function
    
    Args:
        L (int): Maximum path length
        alpha (numpy array): |S|xL backward message variable
        
        with_dummy_state (bool): If true, the final row of the alpha matrix corresponds
            to a dummy state which is used for MDP padding
        
    Returns:
        (float): Partition function value
    """

    # If the dummy state is included, don't include it in the partition
    if with_dummy_state:
        alpha = alpha[0:-1, :]

    return np.sum(alpha[:, 0:L])


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


def marginals(L, t_mat, alpha, beta, Z_theta, gamma=1.0, rsa=None, rsas=None):
    """Compute marginal terms
    
    Args:
        L (int): Maximum path length
        t_mat (numpy array): |S|x|A|x|S| transition matrix
        alpha (numpy array): |S|xL array of backward message values
        beta (numpy array): |S|xL array of forward message values
        Z_theta (float): Partition value
        
        gamma (float): Discount factor
        rsa (numpy array): |S|x|A| array of linear state-action reward weights
        rsas (numpy array): |S|x|A|x|S| array of linear state-action-state reward
            weights
    
    Returns:
        (numpy array): |S|xL array of state marginals
        (numpy array): |S|x|A|xL array of state-action marginals
        (numpy array): |S|x|A|x|S|xL array of state-action-state marginals
    """

    if rsa is None:
        rsa = np.zeros(np.array(t_mat.shape[0:2]))
    if rsas is None:
        rsas = np.zeros(np.array(t_mat.shape[0:3]))

    pts = np.zeros((t_mat.shape[0], L))
    ptsa = np.zeros((t_mat.shape[0], t_mat.shape[1], L - 1))
    ptsas = np.zeros((t_mat.shape[0], t_mat.shape[1], t_mat.shape[2], L - 1))

    for t in range(L - 1):
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):

                    contrib = alpha[s1, t] * (
                        t_mat[s1, a, s2]
                        * np.exp(
                            gamma ** ((t + 1) - 1) * (rsa[s1, a] + rsas[s1, a, s2])
                        )
                        * beta[s2, L - (t + 1) - 1]
                    )

                    # Add to pts, ptsa, ptsas
                    pts[s1, t] += contrib
                    ptsa[s1, a, t] += contrib
                    ptsas[s1, a, s2, t] += contrib

    # Add final column of pts
    pts[:, L - 1] = alpha[:, L - 1]

    # Remove partition contribution
    pts /= Z_theta
    ptsa /= Z_theta
    ptsas /= Z_theta

    return pts, ptsa, ptsas


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
        rsa = np.zeros(np.array(t_mat.shape[0:2]))
    if rsas is None:
        rsas = np.zeros(np.array(t_mat.shape[0:3]))

    pts = np.zeros((t_mat.shape[0], L))
    ptsa = np.zeros((t_mat.shape[0], t_mat.shape[1], L - 1))
    ptsas = np.zeros((t_mat.shape[0], t_mat.shape[1], t_mat.shape[2], L - 1))

    for t in range(L - 1):

        for s1 in range(t_mat.shape[0]):

            if np.isneginf(alpha_log[s1, t]):
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
                            """We don't iterate over valid children for efficiency reasons
                            therefore we manually skip invalid children to avoid
                            log divide-by-zero errors
                            """
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


def env_marginals(env, L, alpha_log, beta_log, Z_theta_log):
    """Convenience method
    
    Args:
        env (.envs.explicit_env.IExplicitEnv) Environment to solve
        L (int): Max path length
        alpha_log (numpy array): |S|xL array of backward message values in log space
        beta_log (numpy array): |S|xL array of forward message values in log space
        Z_theta_log (float): Partition value in log space
    
    Returns:
        (numpy array): |S|xL array of state marginals in log space
        (numpy array): |S|x|A|xL array of state-action marginals in log space
        (numpy array): |S|x|A|x|S|xL array of state-action-state marginals in log space
    """
    return marginals_log(
        L,
        env.t_mat,
        alpha_log,
        beta_log,
        Z_theta_log,
        gamma=env.gamma,
        rsa=env.state_action_rewards,
        rsas=env.state_action_state_rewards,
    )


def env_solve(env, L, with_dummy_state=True):
    """Convenience method to solve an environment for marginals
    
    Args:
        env (.envs.explicit_env.IExplicitEnv) Environment to solve
        L (int): Max path length
        with_dummy_state (bool): Indicates if the environment has been padded with a
            dummy state and action, or not
    
    Returns:
        (numpy array): |S|xL array of state marginals in log space
        (numpy array): |S|x|A|xL array of state-action marginals in log space
        (numpy array): |S|x|A|x|S|xL array of state-action-state marginals in log space
        (float): Partition value in log space
    """
    alpha_log = env_backward(env, L)
    beta_log = env_forward(env, L)
    Z_theta_log = partition_log(L, alpha_log, with_dummy_state=with_dummy_state)
    pts_log, ptsa_log, ptsas_log = env_marginals(
        env, L, alpha_log, beta_log, Z_theta_log
    )
    return pts_log, ptsa_log, ptsas_log, Z_theta_log


def nll_s(theta_s, env, max_path_length, with_dummy_state, phibar_s, verbose):
    if verbose:
        print(theta_s)
    env._state_rewards = theta_s
    with np.errstate(over="raise"):
        pts_log, _, _, Z_log = env_solve(
            env, max_path_length, with_dummy_state=with_dummy_state
        )
        nll = Z_log - theta_s @ phibar_s
        grad = np.sum(np.exp(pts_log), axis=1) - phibar_s
    return nll, grad


def nll_sa(theta_sa, env, max_path_length, with_dummy_state, phibar_sa, verbose):
    if verbose:
        print(theta_sa)
    env._state_action_rewards = theta_sa.reshape((len(env.states), len(env.actions)))
    with np.errstate(over="raise"):
        _, ptsa_log, _, Z_log = env_solve(
            env, max_path_length, with_dummy_state=with_dummy_state
        )
        nll = Z_log - theta_sa @ phibar_sa.flatten()
        grad = (np.sum(np.exp(ptsa_log), axis=2) - phibar_sa).flatten()
    return nll, grad


def nll_sas(theta_sas, env, max_path_length, with_dummy_state, phibar_sas, verbose):
    if verbose:
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
    return nll, grad


def maxent_irl(
    rollouts, env, rs=True, rsa=False, rsas=False, rbound=(-1.0, 1.0), verbose=False,
):
    """Maximum Entropy IRL
    
    Args:
        rollouts (list): List of [(s, a), (s, a), ..., (s, None)] trajectories
        env (.envs.explicit_env.IExplicitEnv) Environment to solve
        
        rs (bool): Optimize for state rewards?
        rsa (bool): Optimize for state-action rewards?
        rsas (bool): Optimize for state-action-state rewards?
        rbound (float): Minimum and maximum reward weight values
        verbose (bool): Extra logging
    
    Returns:
        (numpy array): State reward weights
        (numpy array): State-action reward weights
        (numpy array): State-action-state reward weights
    """

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

    with_dummy_state = False
    if min_path_length != max_path_length:
        warnings.warn("Paths are of unequal lengths - padding environment")
        env = pad_terminal_mdp(env)
        raise NotImplementedError("Need to add dummy states, actions to paths")
        with_dummy_state = True

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

    theta_s = None
    if rs:
        if verbose:
            print("Optimizing state rewards")
        res = minimize(
            nll_s,
            np.zeros(num_states),
            args=(env, max_path_length, with_dummy_state, phibar_s, verbose),
            method="L-BFGS-B",
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
            args=(env, max_path_length, with_dummy_state, phibar_sa, verbose),
            method="L-BFGS-B",
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
            args=(env, max_path_length, with_dummy_state, phibar_sas, verbose),
            method="L-BFGS-B",
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
