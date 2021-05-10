import gym
import torch
import numpy as np
import torch.nn as nn
import itertools as it
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from scipy.stats import multivariate_normal
from scipy.linalg import null_space

from mdp_extras import FeatureFunction


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


def pi_heuristic(s):
    position, velocity = s
    if velocity < 0:
        # Accelerate to left
        return torch.tensor([-1])
    else:
        # Accelerate to right
        return torch.tensor([+1])


def main():

    env = gym.make("MountainCarContinuous-v0")
    gamma = 0.99
    basis_dim = 5
    phi = MCGaussianBasis(num=basis_dim)

    rpos = (-1.2, 0.6)
    rvel = (-0.07, 0.07)

    # Collect demonstrations from 'expert'
    pi_expert = pi_heuristic
    num_episodes = 10
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
        path.append((s, None))
        dataset.append(path)

    # Behaviour cloning to copy demo data with linear policy
    pi = LinearPolicy(len(phi)).behaviour_clone(dataset, phi, log_interval=20)

    for _ in range(5):
        done = False
        s = env.reset()
        while not done:
            a = pi(torch.tensor(phi(s))).detach().numpy()
            s2, r, done, info = env.step(a)
            s = s2
            # env.render()

    # Get the policy gradient on our demo dataset
    phi_bar = phi.demo_average(dataset, gamma=gamma)
    total_grad = np.zeros(len(phi))
    for p in dataset:
        for s, a in p[:-1]:
            feat = phi(s)
            nabla_log_pi = feat / pi(torch.tensor(feat)).item()
            total_grad += nabla_log_pi
    total_grad /= len(dataset)

    # Take outer product to form jacobian
    jac = total_grad.reshape(-1, 1) @ phi_bar.reshape(1, -1)

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
            pass
            # Need to solve a linear program to find null space
            # The following is from the sigma-girl paper repo
            # import cdd
            # A = np.array(ns)
            # b = np.zeros(A.shape[0]).reshape(-1, 1)
            # mat = cdd.Matrix(np.hstack([b, -A]), number_type="float")
            # mat.rep_type = cdd.RepType.INEQUALITY
            # V = np.array(cdd.Polyhedron(mat).get_generators())[:, 1:]
            # if V.shape[0] != 1 and (V != 0).any():
            #     weights = V
            #     weights = ns @ weights.T
            #     weights /= np.sum(weights)
            #     P = np.dot(jac.T, jac)
            #     loss = np.dot(np.dot(weights.T, P), weights)
            #     print("Done")

    else:
        # Need to solve a quadratic program to find null space
        raise NotImplementedError

    # If it still fails, we run Sigma-GIRL

    assert False

    # # gamma = 0.9998
    # gamma = 1.0
    # fv = np.zeros_like(phi([0, 0]))
    # for p in dataset:
    #     pv, a = zip(*p)
    #     pv = np.array(pv)
    #     discounting = gamma ** np.arange(0, len(pv))
    #     fv += (phi(pv) @ discounting) / len(dataset)
    #     plt.plot(pv[:, 0], pv[:, 1], ".-", color="tab:blue")

    # print(fv)

    # m = 50
    # n = 50
    # x, y = np.meshgrid(np.linspace(*rpos, m), np.linspace(*rvel, n))
    # # z = np.array([np.sum(phi(v)) for v in zip(x.flat, y.flat)]).reshape(m, n)
    # z = np.log(np.array([fv @ phi(v) for v in zip(x.flat, y.flat)]).reshape(m, n))
    # plt.contour(x, y, z)
    #
    # plt.xlim(*rpos)
    # plt.ylim(*rvel)
    # plt.xlabel("Position")
    # plt.ylabel("Velocity")
    # plt.grid()
    # plt.show()

    assert False

    pi = Policy()
    score = 0.0
    print_interval = 1

    early_learner = True

    for n_epi in range(1000):
        s = env.reset()
        done = False

        print(f"Early learning? {early_learner}")

        base_rewards = []
        shaped_rewards = []
        step = 0
        while not done:  # CartPole-v1 forced to terminates at 500 step.
            # prob = pi(torch.from_numpy(s).float())
            prob = pi_heuristic(s)
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, info = env.step(a.item())

            # Do some reward shaping
            base_rewards.append(r)
            if early_learner:
                if done and step < 199:
                    # We reached the goal - turn off the curricula
                    print("Reached goal")
                    early_learner = False

                # Reward larger velocities
                shaping = np.abs(s_prime[1]) * 100
                shaped_rewards.append(shaping)
                r += shaping

            # pi.put_data((r, prob[a]))
            s = s_prime
            score += r
            env.render()
            step += 1

        # pi.train_net()

        # if n_epi % print_interval == 0 and n_epi != 0:
        #     print("# of episode :{}, avg score : {}".format(n_epi, score / print_interval))
        #     score = 0.
        #     print(np.array(base_rewards))
        #     print(np.array(shaped_rewards))
    env.close()


if __name__ == "__main__":
    main()
