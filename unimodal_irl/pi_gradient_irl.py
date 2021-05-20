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

"""

import torch
import numpy as np
import scipy as sp

from mdp_extras import MLPCategoricalPolicy

from mdp_extras import *


def form_jacobian(
    policy,
    phi_fn,
    demos,
    gamma=0.9,
):
    """
    Find Jacobian matrix for a single episode

    TODO ajs 20/May/2021 Optimize this function. This was reverse-engineered from the Sigma-GIRL
    official repo to verify it's correctness. I need to optimize this now for PyTorch.
    E.g. we could use https://github.com/toshas/torch-discounted-cumsum to vastly speed up most of
    this function.

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

    average_jac = []
    for traj in demos:

        # If the trajectory ends with a 'None' action, drop that timestep
        if traj[-1][1] is None:
            traj = traj[:-1]

        # Slice out states and actions
        states, actions = zip(*traj)
        # Convert states to features
        features = [phi_fn(s) for s in states]

        episode_length = steps = len(states)
        feature_dim = len(features[0])

        # discounted reward features computation
        gamma = gamma ** np.arange(episode_length)
        phi = features  # Step x Feature dim
        gamma_tiled = np.tile(gamma, (feature_dim, 1))  # Feature Dim x Steps
        gamma_tiled = gamma_tiled.transpose()  # Steps x Feature Dim
        discounted_phi = phi * gamma_tiled  # Step x Feature Dim
        discounted_phi = np.array(discounted_phi)  # Step X Feature dim
        expected_discounted_phi = discounted_phi.sum(axis=0)  # Feature Dim x 1
        expected_discounted_phi /= 1.0  # Feature Dim x 1

        # Compute the gradient of the logarithm of the policy wrt policy parameters
        gradients = []  # Step x Gradient Dim
        for step in range(steps):
            pi.zero_grad()
            p = pi.log_prob_for_state_action(
                torch.tensor(features[step]),
                torch.tensor(actions[step]),
            )
            p.backward()
            step_gradient = pi.param_gradient().detach().numpy()
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
        average_jac.append(estimated_gradients)

    average_jac = np.mean(average_jac, axis=0)
    return average_jac


def main():

    # Construct env
    from multimodal_irl.envs import ElementWorldEnv, element_world_extras

    env = ElementWorldEnv()
    xtr, phi, rewards = element_world_extras(env)
    reward = rewards[0]

    # Collect dataset of demonstration (s, a) trajectories from expert
    _, q_star = vi(xtr, phi, reward)
    pi_star = OptimalPolicy(q_star, stochastic=True)
    num_rollouts = 200
    demos = pi_star.get_rollouts(env, num_rollouts)

    # Construct BC policy
    pi = MLPCategoricalPolicy(len(phi), len(xtr.actions), hidden_size=1000)

    # Perform behaviour cloning to copy demo data with policy
    pi = pi.behaviour_clone(demos, phi, num_epochs=1000, log_interval=100)

    # Recover policy gradient jacobian from BC policy
    jac = form_jacobian(pi, phi, demos)
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
    reward_weights = best_solution.x

    print("Done")


if __name__ == "__main__":
    main()
