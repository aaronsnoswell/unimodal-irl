"""Implements Exact Maximum Entropy IRL from my thesis"""

import numpy as np
from numba import jit
from numba import types
from numba.typed import Dict, List

from scipy.optimize import minimize


from mdp_extras import (
    Linear,
    Disjoint,
    trajectory_reward,
    DiscreteExplicitExtras,
    DiscreteImplicitExtras,
)


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
def nb_backward_pass_log_deterministic_stateonly(
    p0s, L, parents, rs, gamma=1.0, padded=False
):
    """Compute backward message passing variable in log-space

    This version of the backward pass function makes extra assumptions so we can handle
    some much larger problems
     - Dynamics are deterministic
     - Rewards are state-only

    Args:
        p0s (numpy array): Starting state probabilities
        L (int): Maximum path length
        parents (numpy array): Fixed-size parents array. Rows indices correspond to
            states, and the first X elements of each row contain the parent state IDs
            for that state. Any remaining elements of that row are then -1.
        rs (numpy array): |S| array of linear state reward weights

        gamma (float): Discount factor
        padded (bool): Is this MDP padded? In which case, we need to handle the parents
            array with extra caution (it won't have the auxiliary state/action included)

    Returns:
        (numpy array): |S|xL array of backward message values in log space
    """
    num_states = len(p0s)

    alpha = np.zeros((num_states, L))
    alpha[:, 0] = np.log(p0s) + rs

    # Handle padded MDP - the auxiliary state has every state (including itself)
    # as a parent, but this isn't reflected in the parents fixed size array
    num_aux_states = 1 if padded else 0

    for t in range(L - 1):
        for s2 in range(num_states - num_aux_states):
            # Find maximum value among all parents of s2
            m_t = _NINF
            for s1 in parents[s2, :]:
                if s1 < 0:
                    # s2 has no more parents
                    break
                m_t = max(m_t, alpha[s1, t])
            m_t += (gamma ** (t + 1)) * rs[s2]

            # Compute next column of alpha in log-space
            for s1 in parents[s2, :]:
                if s1 < 0:
                    # s2 has no more parents
                    break
                alpha[s2, t + 1] += 1.0 * np.exp(
                    alpha[s1, t] + (gamma ** (t + 1)) * rs[s2] - m_t
                )
            alpha[s2, t + 1] = m_t + np.log(alpha[s2, t + 1])

        if padded:
            # Handle auxiliary state as a special case - every state (including itself
            # is a parent state)
            s_aux = num_states - 1
            s_aux_parents = list(range(num_states))

            # Find maximum value among all parents of s2
            m_t = _NINF
            for s1 in s_aux_parents:
                m_t = max(m_t, alpha[s1, t])
            m_t += (gamma ** (t + 1)) * rs[s_aux]

            # Compute next column of alpha in log-space
            for s1 in s_aux_parents:
                alpha[s_aux, t + 1] += 1.0 * np.exp(
                    alpha[s1, t] + (gamma ** (t + 1)) * rs[s_aux] - m_t
                )
            alpha[s_aux, t + 1] = m_t + np.log(alpha[s_aux, t + 1])

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


@jit(nopython=True)
def nb_forward_pass_log_deterministic_stateonly(L, children, rs, gamma=1.0):
    """Compute forward message passing variable in log space

    This version of the forward pass function makes extra assumptions so we can handle
    some much larger problems
     - Dynamics are deterministic
     - Rewards are state-only

    Args:
        L (int): Maximum path length
        children (numpy array): Fixed-size children array. Rows indices correspond to
            states, and the first X elements of each row contain the child state IDs
            for that state. Any remaining elements of that row are then -1.
        rs (numpy array): Linear state

        gamma (float): Discount factor

    Returns:
        (numpy array): |S| x L array of forward message values in log space
    """
    num_states = len(children)

    beta = np.zeros((num_states, L))
    beta[:, 0] = gamma ** (L - 1) * rs
    for t in range(L - 1):
        for s1 in range(num_states):

            # Find maximum value among children of s1
            m_t = _NINF
            for s2 in children[s1, :]:
                if s2 < 0:
                    # s1 has no more children
                    break
                m_t = max(m_t, beta[s2, t])
            m_t += gamma ** (L - (t + 1) - 1) * rs[s1]

            # Compute next column of beta in log-space
            for s2 in children[s1, :]:
                if s2 < 0:
                    # s1 has no more children
                    break
                beta[s1, t + 1] += np.exp(
                    gamma ** (L - (t + 1) - 1) * rs[s1] + beta[s2, t] - m_t
                )
            beta[s1, t + 1] = m_t + np.log(beta[s1, t + 1])

    return beta


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


@jit(nopython=True)
def nb_marginals_log_deterministic_stateonly(
    L, children, alpha_log, beta_log, Z_theta_log
):
    """Compute marginal terms

    This version of the marginal function makes extra assumptions so we can handle
    some much larger problems
     - Dynamics are deterministic
     - Rewards are state-only

    Args:
        L (int): Maximum path length
        children (numpy array): Fixed-size children array. Rows indices correspond to
            states, and the first X elements of each row contain the child state IDs
            for that state. Any remaining elements of that row are then -1.
        alpha_log (numpy array): |S|xL array of backward message values in log space
        beta_log (numpy array): |S|xL array of forward message values in log space
        Z_theta_log (float): Partition value in log space

    Returns:
        (numpy array): |S|xL array of state marginals in log space
    """

    num_states = len(children)

    pts = np.zeros((num_states, L))

    for t in range(L - 1):

        for s1 in range(num_states):

            # if np.isneginf(alpha_log[s1, t]):
            if np.exp(alpha_log[s1, t]) == 0:
                # Catch edge case where the backward message value is zero to prevent
                # floating point error
                pts[s1, t] = -np.inf
            else:
                # Compute max value
                m_t = _NINF
                for s2 in children[s1, :]:
                    if s2 < 0:
                        # s1 has no more children
                        break
                    m_t = max(m_t, beta_log[s2, L - (t + 1) - 1])
                m_t += alpha_log[s1, t] - Z_theta_log

                # Compute state marginals in log space
                for s2 in children[s1, :]:
                    if s2 < 0:
                        # s1 has no more children
                        break
                    contrib = np.exp(
                        alpha_log[s1, t]
                        + beta_log[s2, L - (t + 1) - 1]
                        - Z_theta_log
                        - m_t
                    )
                    pts[s1, t] += contrib

                if pts[s1, t] == 0:
                    pts[s1, t] = -np.inf
                else:
                    pts[s1, t] = m_t + np.log(pts[s1, t])

    # Compute final column of pts
    pts[:, L - 1] = alpha_log[:, L - 1] - Z_theta_log

    return pts


def log_partition(L, alpha_log, padded=True):
    """Compute the log partition function

    Args:
        L (int): Maximum path length
        alpha_log (numpy array): |S|xL backward message variable in log space

        padded (bool): If true, the final row of the alpha matrix corresponds
            to a dummy state which is used for MDP padding

    Returns:
        (float): Partition function value
    """

    # If the dummy state is included, don't include it in the partition
    if padded:
        alpha_log = alpha_log[0:-1, :]

    # Find maximum value
    m = np.max(alpha_log[:, 0:L])

    # Compute partition in log space
    return m + np.log(np.sum(np.exp(alpha_log[:, 0:L] - m)))


def maxent_log_likelihood(xtr, phi, reward, rollouts, weights=None):
    """
    Find the average log likelihood of a set of paths under a MaxEnt model

    That is,

    \hat{\ell}(\theta) = \E_{\Data}[ \log p(\tau \mid \theta)

    To get the total log-likelihood of the dataset (i.e. gets larger as you add more
    data), multiply the value returned by this function with len(rollouts).

    To get the total data likelihood, take the exponent of that value.

    Args:
        xtr (mdp_extras.DiscreteExplicitExtras): MDP extras
        phi (mdp_extras.FeatureFunction): Feature function to use with linear reward
            parameters.
        reward (mdp_extras.RewardFunction): Reward function
        rollouts (list): List of rollouts, each a list of (s, a) tuples

        weights (numpy array): Optional vector of path weights for weighted IRL problems

    Returns:
        (float): Average log-likelihood of the paths in rollouts under the given reward
    """
    return np.average(
        maxent_path_logprobs(xtr, phi, reward, rollouts, xtr.is_padded),
        weights=weights,
    )


def maxent_path_logprobs(xtr, phi, reward, rollouts):
    """Efficiently compute log probability of a set of paths

    Args:
        xtr (mdp_extras.DiscreteExplicitExtras): MDP extras
        phi (mdp_extras.FeatureFunction): Feature function to use with linear reward
            parameters.
        reward (mdp_extras.RewardFunction): Reward function
        rollouts (list): List of rollouts, each a list of (s, a) tuples

    Returns:
        (list): List of log-probabilities under a MaxEnt model of paths
    """

    # Find max path length
    if len(rollouts) == 1:
        max_path_length = len(rollouts[0])
    else:
        max_path_length = max(*[len(r) for r in rollouts])

    if isinstance(xtr, DiscreteExplicitExtras):
        # Process tabular MDP

        # Catch float overflow as an error - reward magnitude is too large for
        # exponentiation with this max path length
        with np.errstate(over="raise"):
            alpha_log = nb_backward_pass_log(
                xtr.p0s,
                max_path_length,
                xtr.t_mat,
                xtr.gamma,
                *reward.structured(xtr, phi),
            )

    elif isinstance(xtr, DiscreteImplicitExtras):
        # Handle Implicit dynamics MDP

        # Only supports state features - otherwise we run out of memory
        assert (
            phi.type == phi.Type.OBSERVATION
        ), "For DiscreteImplicit MPDs only state-based rewards are supported"

        # Only supports deterministic transitions
        assert (
            xtr.is_deterministic
        ), "For DiscreteImplicit MPDs only deterministic dynamics are supported"

        rs = np.array([reward(phi(s)) for s in xtr.states])

        # Catch float overflow as an error - reward magnitude is too large for
        # exponentiation with this max path length
        with np.errstate(over="raise"):
            # Compute alpha_log
            alpha_log = nb_backward_pass_log_deterministic_stateonly(
                xtr.p0s,
                max_path_length,
                xtr.parents_fixedsize,
                rs,
                gamma=xtr.gamma,
                padded=xtr.is_padded,
            )
    else:
        raise ValueError(f"Unknown MDP class {xtr}")

    with np.errstate(over="raise"):
        # Compute partition value
        Z_theta_log = log_partition(max_path_length, alpha_log, padded=xtr.is_padded)

    path_log_probs = (
        np.array(
            [
                xtr.path_log_probability(p) + trajectory_reward(xtr, phi, reward, p)
                for p in rollouts
            ]
        )
        - Z_theta_log
    )

    return path_log_probs


def maxent_ml_path(xtr, phi, reward, start, goal, max_path_length):
    """Find the ML path from s1 to sg under a MaxEnt model

    If transitions can inccur +ve rewards te returned paths may contain loops

    NB ajs 14/Jan/2020 The log likelihood of the path that we compute internally
        is fine for doing viterbi ML path inference, but it's not the actual path
        log likelihood - it's missing the partition function, and the gamma time offset
        is incorrect (depending on what start time the Viterbi alg picks).

    Args:
        xtr (DiscreteExplicitExtras): MDP Extras object
        phi (FeatureFunction): MDP Featrure function
        reward (Linear): Linear reward function
        start (int): Starting state
        goal (int): End state
        max_path_length (int): Maximum allowable path length to search

    Returns:
        (list): Maximum Likelihood path from start to goal under the given MaxEnt reward
            function, or None if no path is possible
    """

    rs, rsa, rsas = reward.structured(xtr, phi)

    # We check if the rewards are all \le 0 - this allows a sp
    all_negative_rewards = (
        np.max(rs) <= 0 and np.max(rsa.flat) <= 0 and np.max(rsas.flat) <= 0
    )

    # Initialize an SxA LL Viterbi trellis
    sa_lls = np.zeros((len(xtr.states), len(xtr.actions), max_path_length)) - np.inf
    sa_lls[goal, :, :] = xtr.gamma ** np.arange(max_path_length) * rs[goal]

    # Supress divide by zero - we take logs of many zeroes here
    with np.errstate(divide="ignore"):

        # Walk backward to propagate the maximum LL
        for t in range(max_path_length - 2, -1, -1):

            # Max-Reduce over actions to compute state LLs
            # (it's a max because we get to choose our actions)
            s_lls = np.max(sa_lls, axis=1)

            if not np.any(np.isneginf((s_lls[:, t + 1]))):
                # The backward message has propagated to every state one step in the future
                if all_negative_rewards:
                    # In this context we are after a shortest path
                    # Drop any earlier times from the trellis and early stop the backward pass
                    sa_lls = sa_lls[:, :, t + 1 :]
                    break

            # Sweep end states
            for s2 in xtr.states:

                if np.isneginf(s_lls[s2, t + 1]):
                    # Skip this state - it hasn't been reached by probability messages yet
                    continue

                # Sweep actions
                for a in xtr.actions:

                    # Sweep starting states
                    for s1 in xtr.states:

                        if xtr.terminal_state_mask[s1]:
                            # We can't step forward from terminal states - skip this one
                            continue

                        transition_ll = (
                            xtr.gamma ** (t - 1) * rs[s1]
                            + xtr.gamma ** (t) * rsa[s1, a]
                            + xtr.gamma ** (t) * rsas[s1, a, s2]
                            # + xtr.gamma ** (t + 1) * rs[s2]
                            + np.log(xtr.t_mat[s1, a, s2])
                        )

                        if np.isneginf(transition_ll):
                            # This transition is impossible - skip
                            continue

                        # Store the max because we're after the maximum likelihood path
                        sa_lls[s1, a, t] = max(
                            sa_lls[s1, a, t], transition_ll + s_lls[s2, t + 1]
                        )

    # Max-reduce to get state/action ML trellises for conveience
    s_lls = np.max(sa_lls, axis=1)

    # Identify our starting time
    if np.isneginf(np.max(s_lls[start])):
        # There is no feasible path from s1 to sg less or equal to than max_path_length
        return None
    start_time = np.argmax(s_lls[start, :])

    # Walk forward from start state, start time to re-construct path
    state = start
    time = start_time
    ml_path = []
    while state != goal:
        action = np.argmax(sa_lls[state, :, time])
        ml_path.append((state, action))
        successor_states = [s for (a, s) in xtr.children[state] if a == action]

        # Choose successor state with highest log likelihood at time + 1
        ml = -np.inf
        next_state = None
        for s2 in successor_states:
            s2_ll = s_lls[s2, time + 1]
            if s2_ll >= ml:
                next_state = s2
                ml = s2_ll

        state = next_state
        time = time + 1

    # Add final (goal) state
    ml_path.append((state, None))

    return ml_path


def sw_maxent_irl(x, xtr, phi, phi_bar, max_path_length, nll_only=False):
    """Maximum Entropy IRL using our exact algorithm

    Returns NLL and NLL gradient of the demonstration data under the proposed reward
    parameters x.

    N.b. the computed NLL here doesn't include the contribution from the MDP dynamics
    for each path - this term is independent of the parameter x, so doesn't affect the
    optimization result.

    Args:
        x (numpy array): Current reward function parameter vector estimate
        xtr (mdp_extras.BaseExtras): Extras object for the MDP being
            optimized
        phi (mdp_extras.FeatureFunction): Feature function to use with linear reward
            parameters. We require len(phi) == len(x).
        phi_bar (numpy array): Feature expectation. N.b. if using a weighted feature
            expectation, it is very important to make sure the weights you used
            sum to 1.0!
        max_path_length (int): Maximum path length
        nll_only (bool): If true, only return NLL

    Returns:
        (float): Negative Log Likelihood of a MaxEnt model with x as the reward
            parameters and the given feature expectation
        (numpy array): Downhill gradient of negative log likelihood at the given point

    """

    # Store current argument guess
    r_linear = Linear(x)

    if isinstance(xtr, DiscreteExplicitExtras):
        # Process tabular MDP

        # Explode reward function to indicator arrays
        rs, rsa, rsas = r_linear.structured(xtr, phi)

        # Catch float overflow as an error - reward magnitude is too large for
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
            Z_theta_log = log_partition(
                max_path_length, alpha_log, padded=xtr.is_padded
            )

        # Compute NLL
        nll = Z_theta_log - x @ phi_bar

        if nll_only:
            return nll
        else:

            # Compute gradient
            with np.errstate(over="raise"):

                # Compute forward message
                beta_log = nb_forward_pass_log(
                    max_path_length,
                    xtr.t_mat,
                    gamma=xtr.gamma,
                    rs=rs,
                    rsa=rsa,
                    rsas=rsas,
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

    elif isinstance(xtr, DiscreteImplicitExtras):
        # Handle Implicit dynamics MDP

        # Only supports state features - otherwise we run out of memory
        assert (
            phi.type == phi.Type.OBSERVATION
        ), "For DiscreteImplicit MPDs only state-based rewards are supported"

        # Only supports deterministic transitions
        assert (
            xtr.is_deterministic
        ), "For DiscreteImplicit MPDs only deterministic dynamics are supported"

        rs = np.array([r_linear(phi(s)) for s in xtr.states])

        # Catch float overflow as an error - reward magnitude is too large for
        # exponentiation with this max path length
        with np.errstate(over="raise"):
            # Compute alpha_log
            alpha_log = nb_backward_pass_log_deterministic_stateonly(
                xtr.p0s,
                max_path_length,
                xtr.parents_fixedsize,
                rs,
                gamma=xtr.gamma,
                padded=xtr.is_padded,
            )

            # Compute partition value
            Z_theta_log = log_partition(
                max_path_length, alpha_log, padded=xtr.is_padded
            )

        # Compute NLL
        nll = Z_theta_log - x @ phi_bar

        if nll_only:
            return nll
        else:

            # Compute NLL gradient as well
            with np.errstate(over="raise"):
                # Compute beta_log
                beta_log = nb_forward_pass_log_deterministic_stateonly(
                    max_path_length, xtr.children_fixedsize, rs, gamma=xtr.gamma
                )

                # Compute transition marginals pts_log (not ptsa, ptsas)
                pts_log = nb_marginals_log_deterministic_stateonly(
                    max_path_length,
                    xtr.children_fixedsize,
                    alpha_log,
                    beta_log,
                    Z_theta_log,
                )

            # Compute gradient
            s_counts = np.sum(np.exp(pts_log), axis=-1)
            efv_s = np.sum([s_counts[s] * phi(s) for s in xtr.states], axis=0)
            nll_grad = efv_s - phi_bar
            return nll, nll_grad

    else:
        # Unknown MDP type
        raise ValueError(f"Unknown MDP class {xtr}")


def main():
    """Main function"""

    import gym
    import random
    import itertools as it

    from mdp_extras import vi, OptimalPolicy, Indicator
    from mdp_extras.envs import nchain_extras

    n = 2
    env = gym.make("NChain-v0", n=n)
    xtr, phi, reward_gt = nchain_extras(env, gamma=1.0)

    # Change to a uniform MDP
    xtr._t_mat = np.ones_like(xtr.t_mat) / len(xtr.states)
    xtr._p0s = np.ones(len(xtr.states)) / len(xtr.states)
    reward_gt.theta = np.zeros_like(reward_gt.theta)

    # Change phi to state-based indicator
    phi = Indicator(Indicator.Type.OBSERVATION, xtr)

    # Solve the MDP
    max_path_length = n
    # num_demos = 10000
    # demo_star = []
    # for _ in range(num_demos):
    #     demo = []
    #     # Start state
    #     s = random.choice(xtr.states)
    #     for t in it.count():
    #         a = random.choice(xtr.actions)
    #         demo.append((s, a))
    #         s = random.choice(xtr.states)
    #         if t == max_path_length - 2:
    #             break
    #     demo.append((s, None))
    #     demo_star.append(demo)
    # phi_bar_star = phi.demo_average(demo_star)
    phi_bar_star = np.ones(len(phi)) / len(phi)

    x = np.zeros(len(phi))
    nll = sw_maxent_irl(x, xtr, phi, phi_bar_star, max_path_length, nll_only=True)

    pass


if __name__ == "__main__":
    main()
