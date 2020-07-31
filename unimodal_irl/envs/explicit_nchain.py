"""Overloads the n-Chain environment to make the rewards and dynamics explicit

The n-Chain environment is first described "A Bayesian Framework for Reinforcement
Learning" by Strens, 2000.

Base implementation is here:
https://github.com/openai/gym/blob/master/gym/envs/toy_text/nchain.py
"""

import numpy as np
import interface

from gym.envs.toy_text.nchain import NChainEnv

from unimodal_irl.envs.utils import compute_parents_children
from unimodal_irl.envs.explicit import IExplicitEnv, ExplicitEnvGetters


class ExplicitNChainEnv(
    NChainEnv, ExplicitEnvGetters, interface.implements(IExplicitEnv)
):
    """Explicit n-Chain Environment"""

    human_actions = ["→", "←"]

    # Action constants
    A_FORWARD = 0
    A_BACKWARD = 1

    def __init__(self, *args, **kwargs):
        """C-tor"""

        # Call super-constructor
        super().__init__(*args, **kwargs)

        # Populate IExplicitEnv terms by introspection to the super
        self._states = np.arange(self.observation_space.n)
        self._actions = np.arange(self.action_space.n)
        self._gamma = 1.0
        self._p0s = np.zeros(self.observation_space.n)
        self._p0s[0] = 1.0
        self._state_rewards = None
        self._state_action_rewards = np.zeros(
            (self.observation_space.n, self.action_space.n)
        )
        self._state_action_rewards[:, self.A_BACKWARD] = self.small
        self._state_action_rewards[
            self.observation_space.n - 1, self.A_FORWARD
        ] = self.large
        self._state_action_state_rewards = None
        self._terminal_state_mask = np.zeros(self.observation_space.n)

        # Populate dynamics
        self._t_mat = np.zeros(
            (self.observation_space.n, self.action_space.n, self.observation_space.n)
        )

        # Backward action moves to 0th state if it doesn't fail, forward if it does
        self._t_mat[:, self.A_BACKWARD, 0] = 1.0 - self.slip
        for s1 in self._states:
            self._t_mat[
                s1, self.A_BACKWARD, min(s1 + 1, self.observation_space.n - 1)
            ] = self.slip

        # Forward action moves to next state if it doesn't fail, 0th if it does
        for s1 in self._states:
            self._t_mat[
                s1, self.A_FORWARD, min(s1 + 1, self.observation_space.n - 1)
            ] = (1.0 - self.slip)
        self._t_mat[:, self.A_FORWARD, 0] = self.slip

        self._parents, self._children = compute_parents_children(
            self._t_mat, self._terminal_state_mask
        )

        # Update reward range
        self.reward_range = (
            np.min(self._state_action_rewards.flatten()),
            np.max(self._state_action_rewards.flatten()),
        )


def demo():
    """Demo function"""
    # Constructing tests that it meets IExplicitEnv requirements
    env = ExplicitNChainEnv()
    print(env)

    env.reset()
    print("Stepping backward")
    print(env.step(env.A_BACKWARD))
    print(env.step(env.A_BACKWARD))
    print(env.step(env.A_BACKWARD))
    print(env.step(env.A_BACKWARD))
    print(env.step(env.A_BACKWARD))
    print(env.step(env.A_BACKWARD))

    print("Stepping forward")
    _r = 0
    while _r != env.large:
        res = env.step(env.A_FORWARD)
        _, _r, _, _ = res
        print(res)


if __name__ == "__main__":
    demo()
