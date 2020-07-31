"""Example Linear MDP from my thesis, chapter 2"""

import gym
import interface
import numpy as np


from gym import spaces
from unimodal_irl.envs.explicit import IExplicitEnv
from unimodal_irl.envs.utils import compute_parents_children


class LinearMDPEnv(gym.Env, interface.implements(IExplicitEnv)):
    """Example Linear MDP from my thesis, chapter 2"""

    def __init__(self):
        """C-tor"""
        self.observation_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(1)
        self._p0s = np.zeros(self.observation_space.n)
        self._p0s[0] = 1
        self._t_mat = np.zeros(
            (self.observation_space.n, self.action_space.n, self.observation_space.n)
        )
        self._t_mat[0, 0, 1] = 1
        self._t_mat[1, 0, 2] = 1
        self._t_mat[2, 0, 3] = 1
        self._t_mat[3, 0, 3] = 1
        self._terminal_state_mask = np.zeros(self.observation_space.n)
        self._terminal_state_mask[-1] = 1
        self._gamma = 0.9
        self._state_rewards = np.zeros(self.observation_space.n) - 1
        self._state_rewards[-1] = 1
        self._state_action_rewards = None
        self._state_action_state_rewards = None
        self.state = self.reset()

        # Compute parent and child sets from dynamics
        self._parents, self._children = compute_parents_children(
            self._t_mat, self._terminal_state_mask
        )

    @property
    def states(self):
        """State space iterable"""
        return np.arange(self.observation_space.n)

    @property
    def actions(self):
        """Action space iterable"""
        return np.arange(self.action_space.n)

    @property
    def t_mat(self):
        """Transition dynamics"""
        return self._t_mat

    @property
    def p0s(self):
        """Starting state distribution"""
        return self._p0s

    @property
    def terminal_state_mask(self):
        """Vector indicating terminal states"""
        return self._terminal_state_mask

    @property
    def parents(self):
        """Dict mapping states to (s, a) parents"""
        return self._parents

    @property
    def children(self):
        """Dict mapping states to (a, s') children"""
        return self._children

    @property
    def gamma(self):
        """Discount factor"""
        return self._gamma

    @property
    def state_rewards(self):
        """Linear state reward weights"""
        return self._state_rewards

    @property
    def state_action_rewards(self):
        """Linear state-action reward weights"""
        return self._state_action_rewards

    @property
    def state_action_state_rewards(self):
        """Linear state-action-state reward weights"""
        return self._state_action_state_rewards

    def reset(self):
        self.state = np.random.choice(list(range(self.observation_space.n)), p=self.p0s)
        return self.state

    def step(self, action):
        """Step the MDP"""
        assert self.action_space.contains(action)
        next_state_dist = self.t_mat[self.state, action]
        self.state = np.random.choice(
            list(range(self.observation_space.n)), p=next_state_dist
        )
        reward = self.state_rewards[self.state]
        done = bool(self.terminal_state_mask[self.state])
        return self.state, reward, done, {}


def demo():
    """Demo function"""

    # Constructing the MDP will verify it meets the requirements of IExplicitEnv
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
