"""Implements Sigma-GIRL

Ramponi, Giorgia, et al. "Truly Batch Model-Free Inverse Reinforcement Learning about Multiple Intentions." International Conference on Artificial Intelligence and Statistics. PMLR, 2020.

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
            if V.shape[0] != 1 and (V != 0).any():
                weights = V
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

"""

import torch
import numpy as np
import scipy as sp

from mdp_extras import MLPCategoricalPolicy

from mdp_extras import *


def form_jacobian(policy, phi, demos, gamma=1.0):
    """Form policy value jacobian wrt. reward and policy parameters

    TODO ajs 12/May/2021 Optimize + make faster

    Args:
        policy (Policy object): A policy object providing: .zero_grad() function,
            .log_prob(a, phi(s)) to get log probability of an action given a state
            feature vector, and .param_gradient() providing the accumulated auto-diff
            gradient of each parameter in a single long vector.
        phi (mdp_extras.FeatureFunction): Feature function
        demos (list): List of expert demonstrations, each a list of (s, a) tuples

        gamma (float): Discount factor

    Returns:
        (numpy array): d x q Jacobian matrix, where d = # of policy parameters, q = len(phi)
    """
    jac = []
    policy.zero_grad()
    for reward_dim in range(len(phi)):
        f = torch.zeros(1)
        for demo in demos:
            phi_tau = phi.onpath(demo, gamma=gamma)
            phi_tau_f = phi_tau[reward_dim]
            weight = 1.0 / len(demos) * phi_tau_f
            grad_accum = None
            for t, (s, a) in enumerate(demo[:-1]):
                phi_s = torch.tensor(phi(s))
                if grad_accum is None:
                    grad_accum = policy.log_prob_for_state_action(
                        phi_s, torch.tensor(a)
                    )
                else:
                    grad_accum += policy.log_prob_for_state_action(
                        phi_s, torch.tensor(a)
                    )
            f += weight * grad_accum
        f.backward()
        jac.append(policy.param_gradient().numpy())

    # Transpose to get jacobian of size d x q, where d is the size of the policy parameter,
    # and q is the size of the reward parameter
    jac = np.array(jac).T
    # d, q = jac.shape

    return jac


class LinearGaussianPolicy:
    """A linear policy with diagonal gaussian noise, i.e.

    a = W * s + Normal(0, I * sigma)

    """

    def __init__(self, weights, noise=1.0):
        """C-tor

        Args:
            weights (numpy array): |A|x|S| weight matrix mapping inputs to outputs

            noise (float): Multivariate gaussian diagonal standard deviation scale from the mean action vector
        """
        self.weights = weights  # |A|x|S| weight matrix
        self.output, self.input = self.weights.shape
        self.noise_cov = np.diag(
            np.ones(self.output) * noise
        )  # |A|x|A| covariance matrix

    def act(self, s, stochastic=True):
        """Sample an action

        Args:
            s (numpy array): |S| state vector

            stochastic (bool): If true, sample a stochastic action around the mean, if false, just return the computed
                mean action

        Returns:
            (numpy array): |A| action vector
        """

        s = s.reshape(self.input, 1)  # |S|x1 state vector
        a = self.weights @ s  # |A|x1 action vector

        if stochastic:
            noise = np.random.multivariate_normal(
                np.zeros(self.output),  # Mean vector
                self.noise_cov,  # Covariance matrix
                1,  # Size (# samples)
            ).T  # |A|x1 noise vector
            a += noise  # |A|x1 action vector

        return a

    def compute_gradients(self, s, a):
        s = np.array(s).reshape(self.input, 1)  # |S|x1 state vector
        a = np.array(a).reshape(self.output, 1)  # |A|x1 action vector
        mu = self.weights @ s  # |A|x1 action vector
        err = a - mu  # |A|x1 action error vector
        t2 = err @ s.T  # |A|x|S| action by state matrix
        t1 = np.linalg.inv(self.noise_cov)  # |A|x|A| inverse covariance matrix
        ret = t1 @ t2  # |A|x|S| gradient matrix
        return ret.flatten()  # (|A|x|S|) gradient vector


def episode_jacobian(
    pi,
    states,
    actions,
    features,
    discount_f=0.9,
):
    """
    Find Jacobian matrix for a single episode

    Args:
        pi (class): Policy object, providing a method compute_gradients(s, a) that returns the log gradient of the
            policy for action a given state s
        states (numpy array): State vector at each timestep
        actions (numpy array): Action vector at each timestep
        features (numpy array): reward feature vector at each timestep

        disocunt_f (float): Discount factor for MDP


    Returns:
        (numpy array): Shape is num_episodes, policy_parameter_dimension, reward_parameter_dimension
    """
    episode_length = steps = len(episode_states)
    feature_dim = len(episode_features[0])

    # discounted reward features computation
    gamma = discount_f ** np.arange(episode_length)
    phi = episode_features  # Step x Feature dim
    gamma_tiled = np.tile(gamma, (feature_dim, 1))  # Feature Dim x Steps
    gamma_tiled = gamma_tiled.transpose()  # Steps x Feature Dim
    episode_discounted_phi = phi * gamma_tiled  # Step x Feature Dim
    discounted_phi = np.array(episode_discounted_phi)  # Step X Feature dim
    expected_discounted_phi = discounted_phi.sum(axis=0)  # Feature Dim x 1
    expected_discounted_phi /= 1.0  # Feature Dim x 1
    print("Feature Expectation:", expected_discounted_phi)

    # computing the gradients of the logarithm of the policy wrt policy parameters

    gradients = []  # Step x Gradient Dim
    for step in range(steps):
        step_gradient = pi.compute_gradients(
            episode_states[step], episode_actions[step]
        )  # Gradient Dim is the dimension of the policy parameter vector - (|S|x|A|) flat for Ramponi's LinearGaussianPolicy class
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
    phi = np.transpose(
        rep_discounted_phi, axes=[1, 0, 2]
    )  # Step x Gradient Dim x Feature Dim

    # Combine cumulative gradients with feature counts
    tmp = cum_gradients * phi  # Step x Gradient Dim x Feature Dim

    # Sum over steps to get final gradient tensor
    estimated_gradients = tmp.sum(axis=0)  # Gradient Dim x Feature Dim

    # Esimtated gradients is of shape
    # Gradient Dim x Feature Dim
    return estimated_gradients


def main():

    # |A| x |S| policy weight matrix
    weights = np.array([[1.0, 1.0, 0.0, 1.0]])
    noise = 1.0
    pi = LinearGaussianPolicy(weights=weights, noise=noise)

    # State vector is indicator for which of 4 states we are in, with one dummy indicator
    episode_states = np.array(
        [  # Steps
            [1.0, 0.0, 0.0, 1.0],  # State dim
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )

    episode_actions = np.array(
        [  # Steps
            [1.0],  # Action dim
            [1.0],
            [1.0],
            [0.0],
        ]
    )

    # Reward feature vector is regular / terminal state / dummy feature x 3
    episode_features = np.array(
        [  # Steps
            [1.0, 0.0, 1.0, 1.0, 1.0],  # Feature dimension
            [1.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 1.0],
        ],
    )

    # Gradient Dim x Feature Dim
    grads_1ep = episode_jacobian(pi, episode_states, episode_actions, episode_features)

    # TODO: Find this jacobian term for each epsiode, then take mean over all episodes to get the actual jacobian

    assert False

    # Construct env
    from multimodal_irl.envs import ElementWorldEnv, element_world_extras

    env = ElementWorldEnv()
    xtr, phi, rewards = element_world_extras(env)
    reward = rewards[0]

    # Collect dataset of demonstration (s, a) trajectories from expert
    _, q_star = vi(xtr, phi, reward)
    pi_star = OptimalPolicy(q_star, stochastic=True)
    num_rollouts = 50
    demos = pi_star.get_rollouts(env, num_rollouts)

    # Construct BC policy
    pi = MLPCategoricalPolicy(len(phi), len(xtr.actions), hidden_size=100)

    # Perform behaviour cloning to copy demo data with policy
    pi = pi.behaviour_clone(demos, phi, log_interval=100)

    jac = form_jacobian(pi, phi, demos, xtr.gamma)

    # From the original repo
    estimated_gradients, _ = compute_gradient(
        policy_train,
        X_dataset,
        y_dataset,
        r_dataset,
        dones_dataset,
        EPISODE_LENGTH,
        GAMMA,
        features_idx,
        verbose=args.verbose,
        use_baseline=args.baseline,
        use_mask=args.mask,
        scale_features=args.scale_features,
        filter_gradients=args.filter_gradients,
        normalize_f=normalize_f,
    )
    num_episodes, num_parameters, num_objectives = estimated_gradients.shape[:]
    mean_gradients = np.mean(estimated_gradients, axis=0)
    jac = mean_gradients

    d, q = jac.shape
    p_mat = jac.T @ jac

    # Form loss function for sigam-girl
    loss_fn = lambda w: (w.T @ p_mat) @ w

    # Constrain weights to q-1 simplex
    constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * q

    # Perform many random restarts to find best local minima
    print("Searching for reward weights...")
    evaluations = []
    num_random_initialisations = 100
    while len(evaluations) < num_random_initialisations - 1:
        # Choose random initial guess
        omega0 = np.random.uniform(0, 1, q)
        omega0 = omega0 / np.sum(omega0)

        res = sp.optimize.minimize(
            loss_fn,
            omega0,
            method="SLSQP",
            constraints=constraint,
            bounds=bounds,
            options={"ftol": 1e-8, "disp": False},
        )
        if res.success:
            # If the optimization was successful, save it
            evaluations.append([res, loss_fn(res.x)])
    opt_results, losses = zip(*evaluations)

    best_solution = opt_results[np.argmin(losses)]
    print(best_solution)
    reward_weights = best_solution.x

    print("Done")


if __name__ == "__main__":
    main()
