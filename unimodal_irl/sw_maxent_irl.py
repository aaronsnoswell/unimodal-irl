"""Implements Maximum Entropy IRL from my thesis"""

import copy
import warnings
import numpy as np
from numba import jit
from scipy.optimize import minimize


from mdp_extras import (
    Linear,
    Disjoint,
)

from unimodal_irl.utils import empirical_feature_expectations


# Placeholder for 'negative infinity' which doesn't cause NaN in log-space operations
_NINF = np.finfo(np.float64).min


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
        rsas (numpy array): |S|x|A|x|S| array of linear state-action-state reward weights
    
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
    
    TODO ajs Requires re-factor
    
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

    # If this call fails, check that rewards are all dtype float, not int!
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


def maxent_log_partition(
    env, max_path_length, theta_s, theta_sa, theta_sas, with_dummy_state
):
    """Find Maximum Entropy model log partition value
    
    TODO ajs Requires re-factor
    
    Args:
        env (explicit_env.IExplicitEnv): Environment defining dynamics
        max_path_length (int): Maximum path length
        theta_s (numpy array): |S| array of state reward parameters
        theta_sa (numpy array): |S|x|A| array of state-action reward parameters
        theta_sas (numpy array): |S|x|A|x|S| array of state-action-state reward parameters
        with_dummy_state (bool): True if the environment is padded
    
    Returns:
        (float): Log partition value
    """

    # Copy the environment so we don't modify it
    env = copy.deepcopy(env)

    num_states = len(env.states)
    num_actions = len(env.actions)

    if theta_s is None:
        theta_s = np.zeros(num_states)
    if theta_sa is None:
        theta_sa = np.zeros((num_states, num_actions))
    if theta_sas is None:
        theta_sas = np.zeros((num_states, num_actions, num_states))

    assert theta_s.shape == tuple(
        (env.t_mat.shape[0],)
    ), "Passed state rewards do not match environment state dimension"
    assert theta_sa.shape == tuple(
        env.t_mat.shape[0:2]
    ), "Passed state-action rewards do not match environment state-action dimension(s)"
    assert (
        theta_sas.shape == env.t_mat.shape
    ), "Passed state-action-state rewards do not match environment state-action-state dimension(s)"

    env._state_rewards = theta_s
    env._state_action_rewards = theta_sa
    env._state_action_state_rewards = theta_sas

    with np.errstate(over="raise"):
        _, _, _, Z_log = env_solve(
            env, max_path_length, with_dummy_state=with_dummy_state
        )

    return Z_log


def maxent_log_likelihood(
    rollouts,
    env,
    theta_s=None,
    theta_sa=None,
    theta_sas=None,
    with_dummy_state=False,
    path_weights=None,
):
    """Compute the log-likelihood of demonstrations under a MaxEnt model
    
    TODO ajs Requires re-factor
    
    Args:
        rollouts (list): List of list of state-action tuples
        env (explicit_env.IExplicitEnv): Environment to use
        theta_s (numpy array): State reward weights
        theta_sa (numpy array): State-action reward weights
        theta_sas (numpy array): State-action-state reward weights
        with_dummy_state (bool): Have the MDP and rollouts been padded with a dummy
            state using utils.pad_terminal_mdp()?
        path_weights (numpy array): Optional list of path weights - used for performing
            weighted moment matching.
    
    Returns:
        (float): Log likelihood of trained demonstration data under the given reward
            parameters
    """

    num_states = len(env.states)
    num_actions = len(env.actions)

    # Find max path length
    if len(rollouts) == 1:
        max_path_length = min_path_length = len(rollouts[0])
    else:
        max_path_length = max(*[len(r) for r in rollouts])

    if theta_s is None:
        theta_s = np.zeros(num_states)
    if theta_sa is None:
        theta_sa = np.zeros(num_states * num_actions)
    if theta_sas is None:
        theta_sas = np.zeros(num_states * num_actions * num_states)

    Z_log = maxent_log_partition(
        env, max_path_length, theta_s, theta_sa, theta_sas, with_dummy_state
    )

    # Find discounted feature expectations
    phibar_s, phibar_sa, phibar_sas = empirical_feature_expectations(
        env, rollouts, weights=path_weights
    )

    ll = (
        theta_s @ phibar_s.flatten()
        + theta_sa @ phibar_sa.flatten()
        + theta_sas @ phibar_sas.flatten()
        + np.average(
            [env.path_log_probability(p) for p in rollouts],
            weights=path_weights / np.sum(path_weights),
        )
        - Z_log
    )

    return ll


def r_tau(env, tau):
    """Get discounted reward of a trajectory in an environment
    
    TODO ajs Requires re-factor
    
    Args:
        env (explicit_env.IExplicitEnv): Environment defining rewards and discount
        tau (list): State-action trajectory
    """
    r_tau = 0
    if env.state_rewards is not None:
        for t, (s, _) in enumerate(tau):
            r_tau += (env.gamma ** t) * env.state_rewards[s]
    if env.state_action_rewards is not None:
        for t, (s, a) in enumerate(tau[:-1]):
            r_tau += (env.gamma ** t) * env.state_action_rewards[s, a]
    if env.state_action_state_rewards is not None:
        for t, ((s1, a), (s2, _)) in enumerate(zip(tau[:-2], tau[1:-1])):
            r_tau += (env.gamma ** t) * env.state_action_state_rewards[s1, a, s2]
    return r_tau


def maxent_path_logprobs(
    env, rollouts, theta_s=None, theta_sa=None, theta_sas=None, with_dummy_state=False,
):
    """Find MaxEnt log-probability of each path in a list of paths
    
    TODO ajs Requires re-factor
    
    Args:
        env (explicit_env.IExplicitEnv): Environment defining dynamics
        rollouts (list): List of state-action rollouts
        theta_s (numpy array): |S| array of state reward parameters
        theta_sa (numpy array): |S|x|A| array of state-action reward parameters
        theta_sas (numpy array): |S|x|A|x|S| array of state-action-state reward parameters
    
    Returns:
        (list): List of log-probabilities for each path in rollouts
    """

    env = copy.deepcopy(env)

    num_states = len(env.states)
    num_actions = len(env.actions)

    # Find max path length
    if len(rollouts) == 1:
        max_path_length = min_path_length = len(rollouts[0])
    else:
        max_path_length = max(*[len(r) for r in rollouts])

    if theta_s is None:
        theta_s = np.zeros(num_states)
    if theta_sa is None:
        theta_sa = np.zeros((num_states, num_actions))
    if theta_sas is None:
        theta_sas = np.zeros((num_states, num_actions, num_states))

    Z_log = maxent_log_partition(
        env, max_path_length, theta_s, theta_sa, theta_sas, with_dummy_state
    )

    env._state_rewards = theta_s
    env._state_action_rewards = theta_sa
    env._state_action_state_rewards = theta_sas

    path_log_probs = (
        np.array([env.path_log_probability(r) + r_tau(env, r) for r in rollouts])
        - Z_log
    )
    return path_log_probs


def sw_maxent_irl(
    x, xtr, phi, phi_bar, max_path_length, with_dummy_state, nll_only=False
):
    """Compute NLL and gradient for minimization
    
    Args:
        x (numpy array): Current reward function parameter vector estimate
        xtr (mdp_extras.DiscreteExplicitExtras): Extas object for the MDP being
            optimized
        phi (mdp_extras.FeatureFunction): Feature function to use with linear reward
            parameters. We require len(phi) == len(x).
        phi_bar (numpy array): Discounted expected feature vector from some'
            demonstration dataset
        max_path_length (int): Maximum path length to consider
        with_dummy_state (bool): True if the xtr, phi definitions have been padded with
            a dummy state using unimodal_irl.utils.padding_trick(). This is required
            if the MDP has terminal states.
        nll_only (bool): If true, only return NLL
    
    Returns:
        (float): Negative Log Likelihood of a MaxEnt model with x as the reward
            parameters and the given feature expectation
        (numpy array): Downhill gradient of negative log likelihood at the given point
    
    """
    # Store current argument guess and explode reward function to indicator
    # arrays
    r_linear = Linear(x)
    rs, rsa, rsas = r_linear.structured(xtr, phi)

    # Catch float overflow as an error - reward magnitute is too large for
    # exponentiation with this max path length
    with np.errstate(over="raise"):

        # Compute backward message
        alpha_log = nb_backward_pass_log(
            xtr.p0s,
            max_path_length,
            xtr.t_mat,
            gamma=xtr.gamma,
            rs=rs,
            rsa=rsa,
            rsas=rsas,
        )

        # Compute partition value
        Z_theta_log = partition_log(
            max_path_length, alpha_log, with_dummy_state=with_dummy_state
        )

    # Compute NLL
    nll = Z_theta_log - x @ phi_bar

    if nll_only:

        return nll

    else:

        with np.errstate(over="raise"):

            # Compute forward message
            beta_log = nb_forward_pass_log(
                max_path_length, xtr.t_mat, gamma=xtr.gamma, rs=rs, rsa=rsa, rsas=rsas,
            )

            # Compute transition marginals
            pts_log, ptsa_log, ptsas_log = nb_marginals_log(
                max_path_length,
                xtr.t_mat,
                alpha_log,
                beta_log,
                Z_theta_log,
                gamma=xtr.gamma,
                rsa=rsa,
                rsas=rsas,
            )

        # Compute gradient based on feature type
        if phi.type == Disjoint.Type.OBSERVATION:

            s_counts = np.sum(np.exp(pts_log), axis=-1)
            efv_s = np.sum([s_counts[s] * phi(s) for s in xtr.states], axis=0)
            nll_grad = efv_s - phi_bar

        elif phi.type == Disjoint.Type.OBSERVATION_ACTION:

            sa_counts = np.sum(np.exp(ptsa_log), axis=-1)
            efv_sa = np.sum(
                [
                    sa_counts[s1, a] * phi(s1, a)
                    for s1 in xtr.states
                    for a in xtr.actions
                ],
                axis=0,
            )
            nll_grad = efv_sa - phi_bar

        elif phi.type == Disjoint.Type.OBSERVATION_ACTION_OBSERVATION:

            sas_counts = np.sum(np.exp(ptsas_log), axis=-1)
            efv_sas = np.sum(
                [
                    sas_counts[s1, a, s2] * phi(s1, a, s2)
                    for s1 in xtr.states
                    for a in xtr.actions
                    for s2 in xtr.states
                ],
                axis=0,
            )
            nll_grad = efv_sas - phi_bar

        else:
            raise ValueError

        return nll, nll_grad


def main():
    """Main function"""
    pass


if __name__ == "__main__":
    main()
