"""Implements Maximum Entropy IRL from Ziebart 2008, 2010"""

import numpy as np

from numba import jit

from mdp_extras import Linear, Disjoint
from unimodal_irl.sw_maxent_irl import nb_backward_pass_log, log_partition


@jit(nopython=True)
def nb_state_marginals_08_log(p0s, t_mat, rs, max_path_length):
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
    Z_s_log = np.zeros(t_mat.shape[0])
    Z_a_log = np.zeros(t_mat.shape[1]) - np.inf

    # Compute local partition values (Step 2)
    for t in range(max_path_length):

        # Sweep actions
        # Find max value
        m_t = -np.inf
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    m_t = max(m_t, np.log(t_mat[s1, a, s2]) + rs[s1] + Z_s_log[s2])

        Z_a_log[:] = 0
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    Z_a_log[a] += t_mat[s1, a, s2] * np.exp(rs[s1] + Z_s_log[s2] - m_t)
        Z_a_log = m_t + np.log(Z_a_log)

        # Sweep states
        # Find max value
        m_t = np.max(Z_a_log)
        Z_s_log[:] = 0
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    Z_s_log[s1] += np.exp(Z_a_log[a] - m_t)
        Z_s_log = m_t + np.log(Z_s_log)

    # Compute local action probabilities (Step 3)
    prob_sa = np.zeros(t_mat.shape[0 : 1 + 1])
    for s1 in range(t_mat.shape[0]):
        for a in range(t_mat.shape[1]):
            if Z_s_log[s1] == -np.inf:
                # This state was never reached during the backward pass
                # Default to a uniform distribution over actions
                prob_sa[s1, :] = 1.0 / t_mat.shape[1]
            else:
                prob_sa[s1, a] = np.exp(Z_a_log[a] - Z_s_log[s1])

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
def nb_state_marginals_10_log(p0s, t_mat, terminal_state_mask, rs, max_path_length):
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

    # Prepare state and action partition arrays (Step 1)
    # NB: This step is different to the '08 version
    Z_s_log = np.zeros(t_mat.shape[0]) - np.inf
    Z_a_log = np.zeros(t_mat.shape[1]) - np.inf
    Z_s_log[terminal_state_mask] = 0

    # Compute local partition values (Step 2)
    for t in range(max_path_length):

        # Sweep actions
        # Find max value
        m_t = -np.inf
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    m_t = max(m_t, np.log(t_mat[s1, a, s2]) + rs[s1] + Z_s_log[s2])

        Z_a_log[:] = 0
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    Z_a_log[a] += t_mat[s1, a, s2] * np.exp(rs[s1] + Z_s_log[s2] - m_t)
        Z_a_log = m_t + np.log(Z_a_log)

        # Sweep states
        # NB: This step is different to the '08 version
        # Find max value
        m_t = np.max(Z_a_log)
        m_t = max(m_t, 0)
        Z_s_log[:] = 0
        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    if t_mat[s1, a, s2] == 0:
                        continue
                    Z_s_log[s1] += np.exp(Z_a_log[a] - m_t)
                    if terminal_state_mask[s1]:
                        Z_s_log[s1] += np.exp(-m_t)
        Z_s_log = m_t + np.log(Z_s_log)

    # Compute local action probabilities (Step 3)
    prob_sa = np.zeros(t_mat.shape[0 : 1 + 1])
    for s1 in range(t_mat.shape[0]):
        for a in range(t_mat.shape[1]):
            if Z_s[s1] == 0.0:
                # This state was never reached during the backward pass
                # Default to a uniform distribution over actions
                prob_sa[s1, :] = 1.0 / t_mat.shape[1]
            else:
                prob_sa[s1, a] = np.exp(Z_a_log[a] - Z_s_log[s1])

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


def zb_maxent_irl(x, xtr, phi, phi_bar, max_path_length, version="10", nll_only=False):
    """Maximum Entropy IRL using one of Ziebart's approximate algorithms
    
    Returns NLL and NLL gradient of the demonstration data under the proposed reward
    parameters x.
    
    N.b. the computed NLL here doesn't include the contribution from the MDP dynamics
    for each path - this term is independent of the parameter x, so doesn't affect the
    optimization result.
    
    Args:
        x (numpy array): Current reward function parameter vector estimate
        xtr (mdp_extras.DiscreteExplicitExtras): Extras object for the MDP being
            optimized
        phi (mdp_extras.FeatureFunction): Feature function to use with linear reward
            parameters. We require len(phi) == len(x).
        phi_bar (numpy array): Discounted expected feature vector from some
            demonstration dataset
        max_path_length (int): Maximum path length to consider
        
        version (str): Algorithm version to use, one of '08' or '10'
        nll_only (bool): If true, only return NLL
    
    Returns:
        (float): Negative Log Likelihood of a MaxEnt model with x as the reward
            parameters and the given feature expectation
        (numpy array): Downhill gradient of negative log likelihood at the given point
    
    """

    assert (
        phi.type == Disjoint.Type.OBSERVATION
    ), "Ziebart's algorithms only support observation-based features"

    assert version in ["08", "10"], f"Unknown option for 'versoin': {version}"

    # Store current argument guess and explode reward function to indicator
    # arrays
    r_linear = Linear(x)
    rs, _, _ = r_linear.structured(xtr, phi)

    # Catch float overflow as an error - reward magnitute is too large for
    # exponentiation with this max path length
    with np.errstate(over="raise"):

        # Compute backward message
        alpha_log = nb_backward_pass_log(
            xtr.p0s, max_path_length, xtr.t_mat, gamma=xtr.gamma, rs=rs
        )

        # Compute partition value
        Z_theta_log = log_partition(max_path_length, alpha_log, padded=False)

    # Compute NLL
    nll = Z_theta_log - x @ phi_bar

    if nll_only:
        return nll
    else:

        # Compute gradient
        with np.errstate(over="raise"):

            # We use one of Ziebart's approximate algorithms to find the gradient
            if version == "08":
                # Compute state marginals
                s_counts = np.exp(
                    nb_state_marginals_08_log(xtr.p0s, xtr.t_mat, rs, max_path_length)
                )
            elif version == "10":
                s_counts = np.exp(
                    nb_state_marginals_10_log(
                        xtr.p0s,
                        xtr.t_mat,
                        xtr.terminal_state_mask,
                        xtr.state_rewards,
                        max_path_length,
                    )
                )
            else:
                raise ValueError(f"Version must be one of '08' or '10', was {version}")

            efv_s = np.sum([s_counts[s] * phi(s) for s in xtr.states], axis=0)
            nll_grad = efv_s - phi_bar

    return nll, nll_grad
