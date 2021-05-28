"""Implements Sigma-GIRL

Ramponi, Giorgia, et al. "Truly Batch Model-Free Inverse Reinforcement Learning about Multiple Intentions."
International Conference on Artificial Intelligence and Statistics. PMLR, 2020.

Original implementation of PGIRL and Sigma-GIRL is at https://github.com/gioramponi/sigma-girl-MIIRL,
which roughly transliterated is as follows

    import cdd

    import numpy as np
    from qpsolvers import solve_qp
    from scipy.linalg import null_space

    # P-Girl implementation below
    # Solve for jacobian null space
    # jac_rank = np.linalg.matrix_rank(jac)
    ns = null_space(jac)

    # Do we have an effective null space?
    if ns.shape[1] > 0:
        # Do we have an all positive or all negative null space?
        if (ns >= 0).all() or (ns <= 0).all():
            weights = ns[:, 0] / np.sum(ns[:, 0])
            P = np.dot(jac.T, jac)
            loss = np.dot(np.dot(weights.T, P), weights)
            print("Done")
        else:
            # Need to solve a linear program to find null space
            A = np.array(ns)
            b = np.zeros(A.shape[0]).reshape(-1, 1)
            mat = cdd.Matrix(np.hstack([b, -A]), number_type="float")
            mat.rep_type = cdd.RepType.INEQUALITY
            V = np.array(cdd.Polyhedron(mat).get_generators())[:, 1:]
            if V.shape[0] != 1:
                weights = V[1]
                if (weights != 0).any():
                    weights = V[1]
                    weights = ns @ weights.T
                    weights /= np.sum(weights)
                    P = np.dot(jac.T, jac)
                    loss = np.dot(np.dot(weights.T, P), weights)
                    print("Done")
    else:
        # Need to solve a quadratic program to find null space
        raise NotImplementedError

    # If we still fail, we need to try solving a Quadratic Program...
    P = np.dot(jac.T, jac)

    reward_dim = len(phi)
    q = np.zeros(reward_dim)
    A = np.ones((reward_dim, reward_dim))
    b = np.ones(reward_dim)
    G = np.diag(np.diag(A))
    h = np.zeros(reward_dim)

    try:
        weights = solve_qp(P, q, -G, h, A=A, b=b, solver="quadprog")
    except ValueError:
        try:
            normalized_P = P / np.linalg.norm(P)
            weights = solve_qp(normalized_P, q, -G, h, A=A, b=b, solver="quadprog")
        except:
            # If we've still failed to get a reward vector, NOW try Sigma-GIRL

    # Here we try the rank-approxwimate SigmaGIRL implementation, which we have included below

"""

import torch
import numpy as np
import scipy as sp

from sklearn.covariance import ledoit_wolf, oas

from mdp_extras import MLPCategoricalPolicy

from mdp_extras import *


def traj_jacobian(policy, phi, traj, gamma=0.9):
    """Find Sigma-GIRL Empirical Mean and Covariance Jacobian matrices for a single demonstration trajectory

    TODO ajs 20/May/2021 Optimize this function. This was reverse-engineered from the official Sigma-GIRL
        repo to verify it's correctness. I need to optimize this now for PyTorch. E.g. we could use
        https://github.com/toshas/torch-discounted-cumsum to vastly speed up large portions of this function.

    Args:
        policy (mdp_extras.TorchPolicy): Pre-trained policy object. In practice, this is trained with behaviour cloning
            to replicate the demonstration data.
        phi (callable): mdp_extras.FeatureFunction for this MDP
        traj (list): A demonstration trajectory, a list of (s, a) tuples

        gamma (float): Discount factor for MDP

    Returns:
        (numpy array): dxq empirical jacobian mean matrix, where d is the policy parameter dimension and q is the
            reward parameter dimension.
        (numpy array): (dq)x(dq) empirical jacobian covariacne matrix, where d is the policy parameter dimension
            and q is the reward parameter dimension.
        (numpy array): qxq 'P' matrix, q is the reward parameter dimension. Equals jac^T @ jac
    """

    # If the trajectory ends with a 'None' action, drop that timestep
    if traj[-1][1] is None:
        traj = traj[:-1]

    # Slice out states and actions
    states, actions = zip(*traj)

    # Convert states to features
    # TODO ajs 25/May/21 Support (s,a) and (s,a,s') features
    step_features = np.array([phi(s) for s in states])  # Step x Feature dim

    episode_length = steps = len(states)
    feature_dim = len(phi)

    # discounted reward features computation
    step_discount = gamma ** np.arange(episode_length)
    step_discount_tiled = np.tile(
        step_discount, (feature_dim, 1)
    )  # Feature Dim x Steps
    step_discount_tiled = step_discount_tiled.transpose()  # Steps x Feature Dim
    discounted_phi = np.array(step_features * step_discount_tiled)  # Step x Feature Dim
    expected_discounted_phi = discounted_phi.sum(axis=0)  # Feature Dim x 1
    expected_discounted_phi /= 1.0  # Feature Dim x 1

    # Compute the gradient of the logarithm of the policy wrt policy parameters
    gradients = []  # Step x Gradient Dim
    for step in range(steps):
        policy.zero_grad()
        p = policy.log_prob_for_state_action(
            torch.tensor(step_features[step]),
            torch.tensor(actions[step]),
        )
        p.backward()
        step_gradient = policy.param_gradient().detach().numpy()
        gradients.append(step_gradient)
    gradients = np.array(gradients)  # Step x Gradient Dim

    # Measure gradient dimension
    gradient_dim = gradients.shape[1]

    # Repeat gradients over feature dimension
    rep_gradients = np.tile(
        gradients, (feature_dim, 1, 1)
    )  # Feature Dim x Step x Gradients Dim

    # Shuffle dimensions
    rep_gradients = np.transpose(
        rep_gradients, axes=[1, 2, 0]
    )  # Steps x Gradient Dim x Feature Dim

    # Cumulative sum over timesteps to get cumulative gradients
    cum_gradients = rep_gradients.cumsum(
        axis=0
    )  # Steps (Cumulative) x Gradient Dim x Feature Dim

    # Repeat discounted feature counts over gradient dimensions
    rep_discounted_phi = np.tile(
        discounted_phi, (gradient_dim, 1, 1)
    )  # Gradient Dim x Step x Feature Dim

    # Shuffle dims
    rep_discounted_phi_shuffled = np.transpose(
        rep_discounted_phi, axes=[1, 0, 2]
    )  # Step x Gradient Dim x Feature Dim

    # Combine cumulative gradients with feature counts
    tmp = (
        cum_gradients * rep_discounted_phi_shuffled
    )  # Step x Gradient Dim x Feature Dim

    # Sum over steps to get final gradient tensor
    estimated_gradients = tmp.sum(axis=0)  # Gradient Dim x Feature Dim

    return estimated_gradients


def form_jacobian(policy, phi, rollouts, gamma=0.9, weights=None):
    """Find Sigma-GIRL Empirical Mean and Covariance Jacobian matrices for dataset of demonstration trajectories

    Args:
        policy (mdp_extras.TorchPolicy): Pre-trained policy object, In practice, this is trained with behaviour cloning
            to replicate the demonstration data.
        phi (callable): mdp_extras.FeatureFunction for this MDP
        rollouts (list): List of demonstration trajectories, each a list of (s, a) tuples

        gamma (float): Discount factor for MDP
        weights (numpy array): Optional path weights for weighted IRL problems

    Returns:
        all_jacs (list): List of dxq individual jacobians for each demonstration trajectory
        (numpy array): dxq empirical jacobian mean matrix, where d is the policy parameter dimension and q is the
            reward parameter dimension.
        (numpy array): (dq)x(dq) empirical jacobian covariacne matrix, where d is the policy parameter dimension
            and q is the reward parameter dimension.
        (numpy array): qxq 'P' matrix, q is the reward parameter dimension. Equals jac^T @ jac
    """
    trajectory_jacobians = np.array(
        [traj_jacobian(policy, phi, traj, gamma) for traj in rollouts]
    )  # Num demonstrations x Gradient Dim x Feature Dim

    # Trajectory weightings
    if weights is None:
        weights = np.ones(len(rollouts)) / len(rollouts)

    # Find (weighted) mean matrix
    jac_mean = np.average(trajectory_jacobians, axis=0, weights=weights)

    # Find (weighted) covariance matrix
    trajectory_differences = trajectory_jacobians - jac_mean
    trajectory_covariances = [
        np.outer(diff.flat, diff.flat) for diff in trajectory_differences
    ]
    jac_cov = np.average(trajectory_covariances, axis=0, weights=weights)

    # Compute 'P' matrix
    p_mat = jac_mean.T @ jac_mean

    return trajectory_jacobians, jac_mean, jac_cov, p_mat


def pi_gradient_irl(x, p_mat):
    """Compute the Sigma-GIRL loss function for a given reward vector 'x' with a given P matrix

    N.b. Sigma-GIRL requires that the reward parameters are in a q-1 simplex, i.e. all positive and
    sum to 1. This means that when minimizing this objective function, you should attach some constraints.

    E.g. if using scipy:

    res = sp.optimize.minimize(
        pi_gradient_irl,
        x0,
        method="SLSQP",
        constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1},
        bounds=[(0, 1)] * len(x0),
        args=(p_mat)
    )

    Also note, this rank-approximate Sigma-GIRL algorithm finds a local minima, so many random starting
    points should be tried in an outer layer of the optimization, and the one with the argmin should be
    chosen as the best solution

    Args:
        x (numpy array): Candidate reward parameter of length q
        p_mat (numpy array): qxq 'P' matrix, derived from the data jacobian

    Returns:
        (float): Loss for Sigma-GIRL minimization
    """
    # Nothing fancy to see here
    return x.T @ p_mat @ x


def optimal_jacobian_mean(jac_mean, jac_cov, reward_vec):
    """Solve for the optimal jacobian mean matrix M(\omega)

    This is given by the (un-numbered) equation *after* Eq. 6 in Truuly Batch Model Free IRL about MI

    Args:
        jac_mean (numpy array): dxq Jacobian mean, where d is the policy parameter size and
            q is the reward parameter size - denoted \hat{\nabla_\theta}\psi(\theta) in the paper
        jac_cov (numpy array): dqxdq Jacobian covariance matrix - denoted Sigma in the paper
        reward_vec (numpy array): qx1 Reward parameter vector - denoted \omega in the paper

    Returns:
        (numpy array): Optimal dxq mean jacobian matrix for the given jacobian, sigma and omega
    """
    d, q = jac_mean.shape
    assert len(reward_vec) == q
    assert (
        len(jac_cov.shape) == 2
        and jac_cov.shape[0] == jac_cov.shape[1]
        and jac_cov.shape[0] == d * q
    )

    idq = np.eye(d * q)
    id = np.eye(d)

    # Transpose the kronecker delta to match convention in the paper
    a = np.kron(reward_vec, id).T
    b = np.linalg.pinv(a.T @ jac_cov @ a)
    c = idq - jac_cov @ a @ b @ a.T
    M_flat = c @ jac_mean.flat
    M = np.reshape(M_flat, (d, q))
    return M


def gradient_log_likelihood(xtr, phi, reward, rollouts, weights=None):
    """
    Find the average log likelihood of a set of paths under a Sigma-GIRL model

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
    """
    return np.average(
        gradient_path_logprobs(xtr, phi, reward, rollouts),
        weights=weights,
    )


def gradient_path_logprobs(opt_jac_mean, jac_cov, rollout_jacobians):
    """Efficiently compute the log probability of a set of paths under the Sigma-GIRL model

    This is given by the log of Eq. 4 in Truly Batch Model Free IRL about MI, however the empirical
    mean jacobian is computed for each trajectory individually, therefore n=1.

    Args:
        opt_jac_mean (numpy array): dxq Optimal Jacobian mean matrix, e.g. computed using optimal_jacobian_mean()
        jac_cov (numpy array): (dq)x(dq) Jacobian empirical covariance matrix - n.b. this should not
            have any pre-conditioning applied to it
        rollout_jacobians (list): List of dxq jacobians, one for each demonstration trajectory. E.g.
            these terms can be computed by traj_jacobian()

    Returns:
        (list): List of log-probabilities under a Sigma-GIRL model, ignoring the covariance determinant
            term, which is very hard to estimate accuractely
    """

    # N.b. it is very important that this is the plain empirical covariance, and that we allow singular
    # covariance matricies - Scipy internally does Symmetric Positive Semiedefinite decomposition to
    # condition the distribution appropriately if jac_cov is singular
    jac_dist = sp.stats.multivariate_normal(
        mean=opt_jac_mean.flatten(), cov=jac_cov, allow_singular=True
    )

    traj_lls = np.array(
        [
            jac_dist.logpdf(rollout_jacobian.flatten())
            for rollout_jacobian in rollout_jacobians
        ]
    )
    return traj_lls


def main():
    """Demonstrate Sigma-GIRL on ElementWorld"""

    # Construct env
    from multimodal_irl.envs import ElementWorldEnv, element_world_extras

    num_elements = 2
    env = ElementWorldEnv(num_elements=num_elements)
    xtr, phi, rewards = element_world_extras(env)
    reward = rewards[0]

    # Mirror feature function, reward
    _phi = phi
    _reward = reward
    phi = MirrorWrap(_phi)
    reward = phi.update_reward(_reward)

    # Collect dataset of demonstration (s, a) trajectories from expert
    print("Collecting data")
    _, q_star = vi(xtr, phi, reward)
    pi_star = OptimalPolicy(q_star, stochastic=True)
    num_rollouts = 20
    demos = pi_star.get_rollouts(env, num_rollouts)

    # Construct Behaviour Cloned policy
    print("Behaviour cloning...")
    pi = MLPCategoricalPolicy(
        len(phi), len(xtr.actions), hidden_size=30
    ).behaviour_clone(demos, phi, num_epochs=500, log_interval=50)

    print("Policy 1")
    print(
        np.array(
            [
                env.ACTION_SYMBOLS_A2SYM[int(pi.predict((phi(s)), stoch=False)[0])]
                for s in xtr.states
            ]
        ).reshape(-1, 6)
    )

    # Recover policy gradient jacobian from BC policy
    all_jacs, jac_mean, jac_cov, p_mat = form_jacobian(pi, phi, demos)
    d, q = jac_mean.shape

    # Perform many random restarts to find best local minima
    print("Solving for reward weights...")
    evaluations = []
    num_random_initialisations = 1000
    while len(evaluations) < num_random_initialisations - 1:
        # Choose random initial guess
        x0 = np.random.uniform(0, 1, q)
        x0 = x0 / np.sum(x0)

        res = sp.optimize.minimize(
            pi_gradient_irl,
            x0,
            method="SLSQP",
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
            bounds=[(0, 1)] * len(x0),
            args=(p_mat),
            options={"ftol": 1e-8, "disp": False},
        )
        if res.success:
            # If the optimization was successful, save it
            evaluations.append([res.x, res.fun])
    params, losses = zip(*evaluations)
    reward_weights = params[np.argmin(losses)]

    # Un-mirror learned reward weights
    learned_reward = phi.unupdate_reward(Linear(reward_weights))

    # Compute optimal jacobian mean
    opt_jac_mean = optimal_jacobian_mean(jac_mean, jac_cov, reward_weights)

    # Compute log-likelihood of each demo under the learned reward
    rollout_jacobians = np.array([traj_jacobian(pi, phi, d, xtr.gamma) for d in demos])
    traj_lls = gradient_path_logprobs(opt_jac_mean, jac_cov, rollout_jacobians)

    print("GT Reward")
    print(_reward.theta)

    print("Learned reward")
    print(learned_reward.theta)

    from unimodal_irl.metrics import ile_evd

    _, evd = ile_evd(xtr, _phi, _reward, learned_reward)
    print("EVD", evd)

    print("Done")


if __name__ == "__main__":
    main()
