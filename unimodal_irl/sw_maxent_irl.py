import copy
import numpy as np


from unimodal_irl.utils import pad_mdp, compute_parents_children


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
        rs (numpy array): Linear state reward weights
        rsa (numpy array): Linear state-action reward weights
        rsas (numpy array): Linear state-action-state reward weights
    
    Returns:
        (numpy array): |S| x L array of backward message values
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
        rs (numpy array): Linear state reward weights
        rsa (numpy array): Linear state-action reward weights
        rsas (numpy array): Linear state-action-state reward weights
    
    Returns:
        (numpy array): |S| x L array of backward message values in log space
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


def forward_pass(L, t_mat, children, gamma=1.0, rs=None, rsa=None, rsas=None):
    """Compute forward message passing variable
    
    Args:
        L (int): Maximum path length
        t_mat (numpy array): |S|x|A|x|S| transition matrix
        children (dict): Dictionary mapping states to (a, s') child tuples
        gamma (float): Discount factor
        rs (numpy array): Linear state reward weights
        rsa (numpy array): Linear state-action reward weights
        rsas (numpy array): Linear state-action-state reward weights
    
    Returns:
        (numpy array): |S| x L array of forward message values
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
    """Compute marginal terms"""

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


def main():
    """Main function"""

    from unimodal_irl.envs.linear_mdp import LinearMDPEnv

    print("Testing Algorithm with Linear MDP example")

    np.random.seed(0)

    env = LinearMDPEnv()
    reward_scale = 1
    env.state_rewards = np.random.randn(4) * reward_scale
    env.state_action_rewards = np.random.randn(4, 1) * reward_scale
    env.state_action_state_rewards = np.random.randn(4, 1, 4) * reward_scale
    env = pad_mdp(env)
    L = 6
    gamma = 0.9

    # Compute ground truth values for testing

    # Path likelihoods
    ltau1 = np.exp(env.state_rewards[0])
    ltau2 = np.exp(
        env.state_rewards[0]
        + env.state_action_rewards[0, 0]
        + env.state_action_state_rewards[0, 0, 1]
        + gamma * (env.state_rewards[1])
    )
    ltau3 = np.exp(
        env.state_rewards[0]
        + env.state_action_rewards[0, 0]
        + env.state_action_state_rewards[0, 0, 1]
        + gamma
        * (
            env.state_rewards[1]
            + env.state_action_rewards[1, 0]
            + env.state_action_state_rewards[1, 0, 2]
            + gamma * (env.state_rewards[2])
        )
    )
    ltau4 = np.exp(
        env.state_rewards[0]
        + env.state_action_rewards[0, 0]
        + env.state_action_state_rewards[0, 0, 1]
        + gamma
        * (
            env.state_rewards[1]
            + env.state_action_rewards[1, 0]
            + env.state_action_state_rewards[1, 0, 2]
            + gamma
            * (
                env.state_rewards[2]
                + env.state_action_rewards[2, 0]
                + env.state_action_state_rewards[2, 0, 3]
                + gamma * (env.state_rewards[3])
            )
        )
    )

    # Partition value
    z_GT = ltau1 + ltau2 + ltau3 + ltau4

    # Path probabilities
    ptau1_GT = ltau1 / z_GT
    ptau2_GT = ltau2 / z_GT
    ptau3_GT = ltau3 / z_GT
    ptau4_GT = ltau4 / z_GT

    # State marginals
    pts_GT = np.zeros((4, 4))
    pts_GT[0, 0] = ptau1_GT + ptau2_GT + ptau3_GT + ptau4_GT
    pts_GT[1, 1] = ptau2_GT + ptau3_GT + ptau4_GT
    pts_GT[2, 2] = ptau3_GT + ptau4_GT
    pts_GT[3, 3] = ptau4_GT

    # State-action marginals
    ptsa_GT = np.zeros((4, 1, 3))
    ptsa_GT[0, 0, 0] = ptau2_GT + ptau3_GT + ptau4_GT
    ptsa_GT[1, 0, 1] = ptau3_GT + ptau4_GT
    ptsa_GT[2, 0, 2] = ptau4_GT

    ptsas_GT = np.zeros((4, 1, 4, 3))
    ptsas_GT[0, 0, 1, 0] = ptau2_GT + ptau3_GT + ptau4_GT
    ptsas_GT[1, 0, 2, 1] = ptau3_GT + ptau4_GT
    ptsas_GT[2, 0, 3, 2] = ptau4_GT

    # Compute children and parent dictionaries
    parents, children = compute_parents_children(env.t, env.terminal_state_mask)

    alpha = backward_pass(
        env.p0s,
        L,
        env.t,
        parents,
        gamma=gamma,
        rs=env.state_rewards,
        rsa=env.state_action_rewards,
        rsas=env.state_action_state_rewards,
    )

    alpha_log = backward_pass_log(
        env.p0s,
        L,
        env.t,
        parents,
        gamma=gamma,
        rs=env.state_rewards,
        rsa=env.state_action_rewards,
        rsas=env.state_action_state_rewards,
    )

    beta = forward_pass(
        L,
        env.t,
        children,
        gamma=gamma,
        rs=env.state_rewards,
        rsa=env.state_action_rewards,
        rsas=env.state_action_state_rewards,
    )

    beta_log = forward_pass_log(
        L,
        env.t,
        children,
        gamma=gamma,
        rs=env.state_rewards,
        rsa=env.state_action_rewards,
        rsas=env.state_action_state_rewards,
    )

    Z_theta = partition(L, alpha)
    Z_theta_log = partition_log(L, alpha_log)

    pts, ptsa, ptsas = marginals(
        L,
        env.t,
        alpha,
        beta,
        Z_theta,
        gamma,
        rsa=env.state_action_rewards,
        rsas=env.state_action_state_rewards,
    )

    # Drop dummy components
    alpha = alpha[0:-1, 0:4]
    alpha_log = alpha_log[0:-1, 0:4]
    beta = beta[0:-1, 0:4]
    beta_log = beta_log[0:-1, 0:4]
    pts = pts[0:-1, 0:4]
    ptsa = ptsa[0:-1, 0:-1, 0:3]
    ptsas = ptsas[0:-1, 0:-1, 0:-1, 0:3]

    print("Alpha = ")
    print(alpha)
    np.testing.assert_array_almost_equal(alpha, np.exp(alpha_log))

    print("Beta = ")
    print(beta)
    np.testing.assert_array_almost_equal(beta, np.exp(beta_log))

    print("Z = {}".format(Z_theta))
    np.testing.assert_almost_equal(Z_theta, np.exp(Z_theta_log))

    print("p_t(s) = ")
    print(pts)
    np.testing.assert_array_almost_equal(pts, pts_GT)

    print("p_t(s, a) = ")
    print(ptsa)
    np.testing.assert_array_almost_equal(ptsa, ptsa_GT)

    print("p_t(s, a, s') = ")
    print(ptsas)
    np.testing.assert_array_almost_equal(ptsas, ptsas_GT)


if __name__ == "__main__":
    main()
