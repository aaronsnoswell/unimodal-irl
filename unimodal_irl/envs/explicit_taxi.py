"""Overloads the Taxi environment to make the rewards and dynamics explicit

The Taxi environment is first described in "Hierarchical Reinforcement Learning with the
MAXQ Value Function Decomposition" by Erez et al., 2011

The Taxi environment features state-action rewards, and has deterministic dynamics, but
stochastic starting state.

Base implementation is here:
https://github.com/openai/gym/tree/master/gym/envs/toy_text/taxi.py
"""

import numpy as np
import interface

from gym.envs.toy_text.taxi import TaxiEnv

from unimodal_irl.envs.utils import discrete2explicit
from unimodal_irl.envs.explicit import IExplicitEnv, ExplicitEnvGetters


class ExplicitTaxiEnv(TaxiEnv, ExplicitEnvGetters, interface.implements(IExplicitEnv)):
    """Explicit Taxi Environment"""

    reward_range = (-10.0, 20.0)

    human_actions = ["↓", "↑", "←", "→", "P", "D"]

    def __init__(self, *args, **kwargs):
        """C-tor"""

        # Call super-constructor
        super().__init__(*args, **kwargs)

        # Populate IExplicitEnv terms from DiscreteEnv
        discrete2explicit(self)


def demo():
    """Demo function"""
    # Constructing tests that it meets IExplicitEnv requirements
    env = ExplicitTaxiEnv()
    print(env)


if __name__ == "__main__":
    demo()
