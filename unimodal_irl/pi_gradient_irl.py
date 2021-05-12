import gym
import torch
import numpy as np
import scipy as sp
import torch.nn as nn
import itertools as it
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from scipy.stats import multivariate_normal
from scipy.linalg import null_space

from mdp_extras import FeatureFunction

# Range of mountaincar state space
mc_rpos = (-1.2, 0.6)
mc_rvel = (-0.07, 0.07)


class MCGaussianBasis(FeatureFunction):
    """Feature function for MountainCar

    A set of Gaussian functions spanning the state space
    """

    def __init__(self, num=5, pos_range=(-1.2, 0.6), vel_range=(-0.07, 0.07)):
        """C-tor"""
        super().__init__(self.Type.OBSERVATION)

        self.dim = num ** 2
        pos_delta = pos_range[1] - pos_range[0]
        vel_delta = vel_range[1] - vel_range[0]

        pos_mean_diff = pos_delta / (num + 1)
        pos_basis_means = (
            np.linspace(pos_mean_diff * 0.5, pos_delta - pos_mean_diff * 0.5, num)
            + pos_range[0]
        )
        pos_basis_std = pos_mean_diff ** 2 / 10

        vel_mean_diff = vel_delta / (num + 1)
        vel_basis_means = (
            np.linspace(vel_mean_diff * 0.5, vel_delta - vel_mean_diff * 0.5, num)
            + vel_range[0]
        )
        vel_basis_std = vel_mean_diff ** 2 / 10

        covariance = np.diag([pos_basis_std, vel_basis_std])
        means = np.array(list(it.product(pos_basis_means, vel_basis_means)))

        self.rvs = [multivariate_normal(m, covariance) for m in means]

    def __len__(self):
        """Get the length of the feature vector"""
        return self.dim

    def __call__(self, o1, a=None, o2=None):
        """Get feature vector given state(s) and/or action"""
        return np.array([rv.pdf(o1) for rv in self.rvs])


class LinearPolicy(nn.Module):
    """A Linear policy"""

    def __init__(self, f_dim, lr=0.02):
        """C-tor"""
        super(LinearPolicy, self).__init__()
        self.lr = lr

        self.fc1 = nn.Linear(f_dim, 1, bias=False)

        # self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.fc1(x.float())
        return x

    def act(self, x):
        return self(x)

    def behaviour_clone(self, dataset, phi, num_epochs=1000, log_interval=None):
        """Behaviour cloning using full-batch gradient descent

        Args:
            dataset (list): List of (s, a) rollouts to clone from
            phi (FeatureFunction): Feature function accepting states and outputting feature vectors

            num_epochs (int): Number of epochs to train for
            log_interval (int): Logging interval, set to 0 to do no logging
        """
        # Convert states to features, and flatten dataset
        phis = []
        actions = []
        for path in dataset:
            _states, _actions = zip(*path[:-1])
            phis.extend([phi(s) for s in _states])
            actions.extend(_actions)
        phis = torch.tensor(phis)
        actions = torch.tensor(actions)

        for epoch in range(num_epochs):
            # Run one epoch of training
            self.optimizer.zero_grad()
            # loss = self.loss_fn(self(phis), actions)
            loss = torch.mean(torch.norm(self(phis) - actions, dim=0) ** 2)
            loss.backward()
            self.optimizer.step()

            if log_interval is not None and epoch % log_interval == 0:
                print(f"Epoch {epoch}, loss={loss.item()}")

        return self


def mc_heuristic_policy(s):
    position, velocity = s
    if velocity < 0:
        # Accelerate to left
        return torch.tensor([-1])
    else:
        # Accelerate to right
        return torch.tensor([+1])


def solve_girl_approx(P, seed=None, verbose=False, num_random_retries=100):
    num_objectives = P.shape[0]

    def quad(w):
        obj = np.dot(np.dot(w.T, P), w)
        return obj

    obj_func = quad
    constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bound = (0, 1)
    bounds = [bound] * num_objectives
    if seed is not None:
        np.random.seed(seed)
    evaluations = []
    i = 0

    while i < num_random_retries:
        x0 = np.random.uniform(0, 1, num_objectives)
        x0 = x0 / np.sum(x0)
        res = sp.optimize.minimize(
            obj_func,
            x0,
            method="SLSQP",
            constraints=constraint,
            bounds=bounds,
            options={"ftol": 1e-8, "disp": verbose},
        )
        if res.success:
            evaluations.append([res.x, obj_func(res.x)])
            i += 1
    solns, losses = zip(*evaluations)
    print("Losses were:")
    print(losses)
    evaluations = np.array(evaluations)
    min_index = np.argmin(evaluations[:, 1])
    x, y = evaluations[min_index, :]
    return x, y


def pi_sigma_girl(P, seed=None, verbose=False, num_random_retries=100):
    num_objectives = P.shape[0]

    def quad(w):
        obj = np.dot(np.dot(w.T, P), w)
        return obj

    obj_func = quad
    constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bound = (0, 1)
    bounds = [bound] * num_objectives
    if seed is not None:
        np.random.seed(seed)
    evaluations = []
    i = 0

    while i < num_random_retries:
        x0 = np.random.uniform(0, 1, num_objectives)
        x0 = x0 / np.sum(x0)
        res = sp.optimize.minimize(
            obj_func,
            x0,
            method="SLSQP",
            constraints=constraint,
            bounds=bounds,
            options={"ftol": 1e-8, "disp": verbose},
        )
        if res.success:
            evaluations.append([res.x, obj_func(res.x)])
            i += 1
    solns, losses = zip(*evaluations)
    print("Losses were:")
    print(losses)
    evaluations = np.array(evaluations)
    min_index = np.argmin(evaluations[:, 1])
    x, y = evaluations[min_index, :]
    return x, y


def main():

    visualize = False

    # Construct continuous mountain car env
    print("Building environment")
    gamma = 0.99
    env = gym.make("MountainCarContinuous-v0")

    # Form a gaussian basis feature function with basis_dim x basis_dim gaussians distributed through state-space
    basis_dim = 5
    phi = MCGaussianBasis(num=basis_dim)

    # Collect dataset of demonstration (s, a) trajectories from expert
    pi_expert = mc_heuristic_policy
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
    pi = LinearPolicy(len(phi))

    # Perform behaviour cloning to copy demo data with policy
    print("Behaviour cloning linear policy on expert data")
    pi = pi.behaviour_clone(dataset, phi, log_interval=20)

    if visualize:
        print("Previewing BC-trained linear policy")
        for episode in range(5):
            done = False
            s = env.reset()
            while not done:
                state_feature = phi(s)
                a = pi(torch.tensor(state_feature)).detach().numpy()
                s, r, done, info = env.step(a)
                env.render()
        env.close()

    print("Constructing Jacobian matrix...")
    # Get the feature expectation from the expert dataset
    phi_bar = phi.demo_average(dataset, gamma=gamma)

    # Get the average log policy gradient on our demo dataset
    f = None
    for p in dataset:
        # Compute weight for this gradient contrib
        phi_bar = torch.tensor(phi.onpath(p, gamma=gamma) / len(dataset))
        for s, a in p[:-1]:
            if f is None:
                f = phi_bar * torch.log(pi(torch.tensor(phi(s))))
            else:
                f += phi_bar * torch.log(pi(torch.tensor(phi(s))))
    f.retain_grad()
    f2 = torch.sum(f)
    f2.backward()
    print("Here")

    # Take outer product to form Jacobian matrix
    # Jacobian is of size d x q, where d is the size of the policy parameter,
    # and q is the size of the reward parameter, which in this case are the same
    q = len(phi)
    d = len(phi)
    jac = average_log_grad.reshape(-1, 1) @ phi_bar.reshape(1, -1)

    # Form 'P' matrix, which is of size q x q (perhaps it stands for 'plane' in plane-GIRL?)
    p_mat = jac.T @ jac

    # Form loss function for sigam-girl
    def loss_fn(omega):
        """Find loss for a given reward weight vector omega"""
        # return np.dot(np.dot(omega.T, p_mat), omega)
        return (omega.T @ p_mat) @ omega

    # Constrain weights to q-1 simplex
    constraint = {"type": "eq", "fun": lambda omega: np.sum(omega) - 1}
    bounds = [(0, 1),] * q

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

    assert False

    # P-Girl implementation below
    # # # Solve for jacobian null space
    # # # jac_rank = np.linalg.matrix_rank(jac)
    # # ns = null_space(jac)
    # #
    # # # Do we have an effective null space?
    # # if ns.shape[1] > 0:
    # #     # Do we have an all positive or all negative null space?
    # #     if (ns >= 0).all() or (ns <= 0).all():
    # #         weights = ns[:, 0] / np.sum(ns[:, 0])
    # #         P = np.dot(jac.T, jac)
    # #         loss = np.dot(np.dot(weights.T, P), weights)
    # #         print("Done")
    # #     else:
    # #         pass
    # #         # Need to solve a linear program to find null space
    # #         # The following is from the sigma-girl paper repo
    # #         # import cdd
    # #         # A = np.array(ns)
    # #         # b = np.zeros(A.shape[0]).reshape(-1, 1)
    # #         # mat = cdd.Matrix(np.hstack([b, -A]), number_type="float")
    # #         # mat.rep_type = cdd.RepType.INEQUALITY
    # #         # V = np.array(cdd.Polyhedron(mat).get_generators())[:, 1:]
    # #         # if V.shape[0] != 1 and (V != 0).any():
    # #         #     weights = V
    # #         #     weights = ns @ weights.T
    # #         #     weights /= np.sum(weights)
    # #         #     P = np.dot(jac.T, jac)
    # #         #     loss = np.dot(np.dot(weights.T, P), weights)
    # #         #     print("Done")
    # #
    # # else:
    # #     # Need to solve a quadratic program to find null space
    # #     raise NotImplementedError
    #
    # # If it still fails, we run Sigma-GIRL
    # P = np.dot(jac.T, jac)
    #
    # # num_objectives == Number of reward parameter dimensions ('q' in the paper)
    # # num_objectives = len(phi)
    # # q = np.zeros(num_objectives)
    # # A = np.ones((num_objectives, num_objectives))
    # # b = np.ones(num_objectives)
    # # G = np.diag(np.diag(A))
    # # h = np.zeros(num_objectives)
    # # normalized_P = P / np.linalg.norm(P)
    # # solver = "quadprog"
    # # from qpsolvers import solve_qp
    # #
    # # try:
    # #     weights = solve_qp(P, q, -G, h, A=A, b=b, solver=solver)
    # # except ValueError:
    # #     try:
    # #         weights = solve_qp(normalized_P, q, -G, h, A=A, b=b, solver=solver)
    # #     except:
    # #         # normalize matrix
    # #
    # #         print("Error in Girl")
    # #         # print(P)
    # #         # print(normalized_P)
    # #         # u, s, v = np.linalg.svd(P)
    # #         # print("Singular Values:", s)
    # #         # ns = scipy.linalg.null_space(mean_gradients)
    # #         # print("Null space:", ns)
    # #
    # #         seed = 1234
    # #         weights, loss = solve_girl_approx(P, seed=seed)
    #
    # seed = 1234
    # weights, loss = solve_girl_approx(P, seed=seed)
    # loss = np.dot(np.dot(weights.T, P), weights)

    print("Done")


if __name__ == "__main__":
    main()
