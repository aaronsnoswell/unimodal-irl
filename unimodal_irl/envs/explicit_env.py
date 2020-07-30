"""Defines an interface for an 'Explicit' OpenAI Gym class

This is a slightly more constrained version of gym.Env, whereby the dynamics and rewards
are explicit (in the form of a transition matrix, linear weights etc.)
"""

import interface


class ExplicitEnvGetters:
    """Convenience class that provides getters for protected IExplicitEnv properties"""

    @property
    def states(self):
        """Iterable over MDP states"""
        return self._states

    @property
    def actions(self):
        """Iterable over MDP actions"""
        return self._actions

    @property
    def p0s(self):
        """|S| vector of starting state probabilities"""
        return self._p0s

    @property
    def t_mat(self):
        """|S|x|A|x|S| array of transition probabilities"""
        return self._t_mat

    @property
    def terminal_state_mask(self):
        """|S| vector indicating terminal states"""
        return self._terminal_state_mask

    @property
    def parents(self):
        """Dict mapping a state to it's (s, a) parents"""
        return self._parents

    @property
    def children(self):
        """Dict mapping a state to it's (a, s') children"""
        return self._children

    @property
    def gamma(self):
        """Discount factor"""
        return self._gamma

    @property
    def state_rewards(self):
        """|S| vector of linear state reward weights"""
        return self._state_rewards

    @property
    def state_action_rewards(self):
        """|S|x|A| array of linear state-action reward weights"""
        return self._state_action_rewards

    @property
    def state_action_state_rewards(self):
        """|S|x|A|x|S| array of linear state-action-state reward weights"""
        return self._state_action_state_rewards


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
        pass

    @property
    def state_action_state_rewards(self):
        """|S|x|A|x|S| array of linear state-action-state reward weights"""
        pass
