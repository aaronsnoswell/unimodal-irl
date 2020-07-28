"""Defines an interface for an 'Explicit' OpenAI Gym class

This is a slightly more constrained version of gym.Env, whereby the dynamics and rewards
are explicit (in the form of a transition matrix, linear weights etc.)
"""

import interface


class IExplicitEnv(interface.Interface):
    """An MDP with explicit dynamics and linear reward"""

    @property
    def states(self):
        """Iterable over MDP states"""
        pass

    @property
    def actions(self):
        """Iterable over MDP actions"""
        pass

    @property
    def p0s(self):
        """|S| vector of starting state probabilities"""
        pass

    @property
    def t_mat(self):
        """|S|x|A|x|S| array of transition probabilities"""
        pass

    @property
    def terminal_state_mask(self):
        """|S| vector indicating terminal states"""
        pass

    @property
    def parents(self):
        """Dict mapping a state to it's (s, a) parents"""
        pass

    @property
    def children(self):
        """Dict mapping a state to it's (a, s') children"""
        pass

    @property
    def gamma(self):
        """Discount factor"""
        pass

    @property
    def state_rewards(self):
        """|S| vector of linear state reward weights"""
        pass

    @property
    def state_action_rewards(self):
        """|S|x|A| array of linear state-action reward weights"""
        return None

    @property
    def state_action_state_rewards(self):
        """|S|x|A|x|S| array of linear state-action-state reward weights"""
        return None
