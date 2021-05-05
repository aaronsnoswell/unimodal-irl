import gym
import torch
import numpy as np
import torch.nn as nn
import itertools as it
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from scipy.stats import multivariate_normal

# Hyperparameters
learning_rate = 0.0002


class MCGaussianBasis:
    """Feature function for MountainCar

    A set of Gaussian functions spanning the state space
    """

    def __init__(self, num=5, pos_range=(-1.2, 0.6), vel_range=(-0.07, 0.07)):
        """C-tor"""

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

    def __call__(self, *args, **kwargs):
        """Query the feature function"""
        return np.array([rv.pdf(args[0]) for rv in self.rvs])


class LinearPolicy(nn.Module):
    """A Linear policy"""

    def __init__(self, f_dim, lr=0.02):
        """C-tor"""
        super(LinearPolicy, self).__init__()
        self.lr = lr

        self.fc1 = nn.Linear(f_dim, 3)  # Accelerate to left, No-op, Accelerate to right

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.fc1(x.float())
        return x

    def sample(self, x):
        x = self(x)
        action_probs = F.softmax(x, dim=0)
        return Categorical(action_probs).sample()

    def behaviour_clone(self, dataset, num_epochs=1000, log_interval=20):
        # Flatten dataset
        phis = []
        actions = []
        for path in dataset:
            _phis, _actions = zip(*path[:-1])
            phis.extend(_phis)
            actions.extend(_actions)
        phis = torch.tensor(phis)
        actions = torch.tensor(actions)

        for epoch in range(num_epochs):
            # Run one epoch of training
            self.optimizer.zero_grad()
            loss = self.loss_fn(self(phis), actions)
            loss.backward()
            self.optimizer.step()

            if epoch % log_interval == 0:
                print(f"Epoch {epoch}, loss={loss.item()}")

        return self


def pi_heuristic(s):
    position, velocity = s
    if velocity < 0:
        # Accelerate to left
        return torch.tensor([1.0, 0.0, 0.0])
    else:
        # Accelerate to right
        return torch.tensor([0.0, 0.0, 1.0])


def main():

    env = gym.make("MountainCar-v0")
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
            a = Categorical(pi_expert(s)).sample()
            path.append((phi(s), a.item()))
            s_prime, r, done, info = env.step(a.item())
            s = s_prime
        path.append((phi(s), None))
        dataset.append(path)

    # Behaviour cloning to copy demo data with linear policy
    pi = LinearPolicy(basis_dim ** 2).behaviour_clone(dataset)

    # Get the policy gradient on our demo dataset

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
