import gym
import pyglet
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from gym import spaces


class LinearMDPEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(1)
        self.p0s = np.zeros(self.observation_space.n)
        self.p0s[0] = 1
        self.t = np.zeros(
            (self.observation_space.n, self.action_space.n, self.observation_space.n)
        )
        self.t[0, 0, 1] = 1
        self.t[1, 0, 2] = 1
        self.t[2, 0, 3] = 1
        self.t[3, 0, 3] = 1
        self.state_rewards = np.zeros(self.observation_space.n) - 1
        self.state_rewards[-1] = 1
        self.terminal_state_mask = np.zeros(self.observation_space.n)
        self.terminal_state_mask[-1] = 1
        self.state = self.reset()

    def reset(self):
        self.state = np.random.choice(list(range(self.observation_space.n)), p=self.p0s)
        return self.state

    def step(self, action):
        assert self.action_space.contains(action)
        next_state_dist = self.t[self.state, action]
        self.state = np.random.choice(
            list(range(self.observation_space.n)), p=next_state_dist
        )
        reward = self.state_rewards[self.state]
        done = bool(self.terminal_state_mask[self.state])
        return self.state, reward, done, {}


def demo():
    """Main function"""
    e = LinearMDPEnv()
    e.reset()
    print(e.step(0))
    print(e.step(0))
    print(e.step(0))
    print(e.step(0))
    print(e.step(0))
    print(e.step(0))


if __name__ == "__main__":
    demo()
