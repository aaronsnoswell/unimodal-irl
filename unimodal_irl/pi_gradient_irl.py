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
    solver = "quadprog"

    try:
        weights = solve_qp(P, q, -G, h, A=A, b=b, solver=solver)
    except ValueError:
        try:
            normalized_P = P / np.linalg.norm(P)
            weights = solve_qp(normalized_P, q, -G, h, A=A, b=b, solver=solver)
        except:
            # If we've still failed to get a reward vector, NOW try Sigma-GIRL

"""

import gym
import torch
import numpy as np
import scipy as sp
import itertools as it

from torch.distributions import Categorical


from mdp_extras import FeatureFunction, MLPGaussianPolicy
from mdp_extras.envs.mountain_car import GaussianBasis


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


def main():

    visualize = False

    # Construct continuous mountain car env
    print("Building environment")
    gamma = 0.99
    env = gym.make("MountainCarContinuous-v0")

    # Form a gaussian basis feature function with basis_dim x basis_dim gaussians distributed through state-space
    basis_dim = 4
    phi = GaussianBasis(num=basis_dim)

    # Collect dataset of demonstration (s, a) trajectories from expert
    pi_expert = lambda s: torch.tensor([-1]) if s[1] < 0 else torch.tensor([+1])
    num_episodes = 20
    print(f"Collecting {num_episodes} expert demonstrations")
    dataset = []
    for episode in range(num_episodes):
        path = []
        done = False
        s = env.reset()
        while not done:
            a = pi_expert(s).numpy()
            path.append((s, a))
            s_prime, r, done, info = env.step(a)
            s = s_prime
            if visualize:
                if episode < 5:
                    env.render()
        path.append((s, None))
        dataset.append(path)
    env.close()

    # Construct a policy that is a linear function of the basis features
    pi = MLPGaussianPolicy(len(phi))

    # Perform behaviour cloning to copy demo data with policy
    print("Behaviour cloning linear policy on expert data")
    pi = pi.behaviour_clone(dataset, phi, log_interval=100)

    if visualize:
        print("Previewing BC-trained linear policy")
        with torch.no_grad():
            for episode in range(5):
                done = False
                s = env.reset()
                while not done:
                    phi_s = phi(s)
                    a = pi.sample_action(torch.tensor(phi_s)).detach().numpy()
                    s, r, done, info = env.step(a)
                    env.render()
            env.close()

    jac = form_jacobian(pi, phi, dataset, gamma)
    d, q = jac.shape

    # Form 'P' matrix
    p_mat = jac.T @ jac

    # Form loss function for sigam-girl
    loss_fn = lambda w: (w.T @ p_mat) @ w

    # Constrain weights to q-1 simplex
    constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * q

    # Perform many random restarts to find best local minima
    print("Searching for reward weights...")
    evaluations = []
    i = 0
    num_random_initialisations = 100
    verbose = False
    while i < num_random_initialisations:
        # Choose random initial guess
        omega0 = np.random.uniform(0, 1, q)
        omega0 = omega0 / np.sum(omega0)

        res = sp.optimize.minimize(
            loss_fn,
            omega0,
            method="SLSQP",
            constraints=constraint,
            bounds=bounds,
            options={"ftol": 1e-8, "disp": verbose},
        )
        if res.success:
            # If the optimization was successful, save it
            evaluations.append([res, loss_fn(res.x)])
            i += 1
    opt_results, losses = zip(*evaluations)

    best_solution = opt_results[np.argmin(losses)]
    print(best_solution)

    learned_reward_parameter = best_solution.x

    # Preview learned reward
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(learned_reward_parameter.reshape(basis_dim, basis_dim))
    plt.colorbar()
    plt.grid()
    plt.show()

    print("Done")


if __name__ == "__main__":
    main()
