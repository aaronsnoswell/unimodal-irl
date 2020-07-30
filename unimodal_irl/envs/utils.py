"""Various utility methods related to Gym environments"""

import copy
import numpy as np
import itertools as it
from types import MethodType

import gym
from gym import spaces


def compute_parents_children(t_mat, terminal_state_mask):
    """Compute parent and child dictionaries
    
    Args:
        t_mat (numpy array): |S|x|A|x|S| array of transition dynamics
        terminal_state_mask (numpy array): |S| vector indicating terminal states
    
    Returns:
        (dict): Dictionary mapping states to (s, a) parent tuples
        (dict): Dictionary mapping states to (a, s') child tuples
    """
    parents = {s: [] for s in range(t_mat.shape[0])}
    children = copy.deepcopy(parents)
    for s2 in range(t_mat.shape[0]):
        for s1, a in np.argwhere(t_mat[:, :, s2] > 0):
            if not terminal_state_mask[s1]:
                parents[s2].append(tuple((s1, a)))
                children[s1].append(tuple((a, s2)))

    return parents, children


def pad_terminal_mdp(env):
    """Pads a terminal MDP, adding a dummy state and action
    
    We gain a O(|S|) space and time efficiency improvement with our MaxEnt IRL algorithm
    for MDPs with terminal states by transforming them to have no terminal states. This
    is done by adding a dummy state and action that pad any trajectories out to a fixed
    upper length.
    
    Args:
        env (.explicit_env.IExplicitEnv) Explicit MDP environment
    
    Returns:
        (.explicit_env.IExplicitEnv) Explicit MDP environment, padded with a dummy
            state and action so that it has no terminal states.
    """

    # Add an extra state and action to the dynamics
    t_mat2 = np.pad(env.t_mat, (0, 1), mode="constant")
    p0s2 = np.pad(env.p0s, (0, 1), mode="constant")
    terminal_state_mask2 = np.pad(env.terminal_state_mask, (0, 1), mode="constant")

    state_rewards2 = None
    state_action_rewards2 = None
    state_action_state_rewards2 = None

    # Dummy state is absorbing
    t_mat2[-1, -1, -1] = 1

    # Terminal states are no longer absorbing
    for terminal_state in np.argwhere(env.terminal_state_mask):
        t_mat2[terminal_state, :, terminal_state] = 0
    terminal_state_mask2 = np.zeros(t_mat2.shape[0])

    # Dummy state reachable anywhere if dummy action is taken
    t_mat2[:, -1, -1] = 1

    # Dummy state doesn't modify rewards
    if env.state_rewards is not None:
        state_rewards2 = np.pad(env.state_rewards, (0, 1), mode="constant")
        state_rewards2[-1] = 0
    if env.state_action_rewards is not None:
        state_action_rewards2 = np.pad(
            env.state_action_rewards, (0, 1), mode="constant"
        )
        state_action_rewards2[:, -1] = 0
    if env.state_action_state_rewards is not None:
        state_action_state_rewards2 = np.pad(
            env.state_action_state_rewards, (0, 1), mode="constant"
        )
        state_action_state_rewards2[:, 0:-1, -1] = -np.inf  # Illegal transition
        state_action_state_rewards2[:, -1, -1] = 0

    # Overwrite environment properties
    env._t_mat = t_mat2
    env._p0s = p0s2
    env._terminal_state_mask = terminal_state_mask2
    env.observation_space = spaces.Discrete(env.observation_space.n + 1)
    env.action_space = spaces.Discrete(env.action_space.n + 1)

    # Update parent and children mappings
    env._parents, env._children = compute_parents_children(
        env.t_mat, env.terminal_state_mask
    )

    if env.state_rewards is not None:
        env._state_rewards = state_rewards2
    if env.state_action_rewards is not None:
        env._state_action_rewards = state_action_rewards2
    if env.state_action_state_rewards is not None:
        env._state_action_state_rewards = state_action_state_rewards2

    return env


def discrete2explicit(EnvClass, env, *, gamma=1.0):
    """Make a DiscreteEnv compliant with IExplicitEnv
    
    Args:
        EnvClass (class): Class of the environment we are modifying
        env (gym.envs.toy_text.discrete.DiscreteEnv): Environment to update
        
        gamma (float): Discount factor to assign
    
    Returns:
        (gym.Env): A gym environment that is compatible with IExplicitEnv
    """

    # Work with a copy, not the original
    env = copy.deepcopy(env)

    env._states = np.arange(env.nS)
    setattr(EnvClass, "states", property(lambda self: self._states))
    env._actions = np.arange(env.nA)
    setattr(EnvClass, "actions", property(lambda self: self._actions))
    env._p0s = np.array(env.isd)
    setattr(EnvClass, "p0s", property(lambda self: self._p0s))

    # Build transition dynamics
    env._t_mat = np.zeros((env.nS, env.nA, env.nS))
    env._terminal_state_mask = np.zeros_like(env._states)
    for s1 in env._states:
        for a in env._actions:
            for prob, s2, r, done in env.P[s1][a]:
                env._t_mat[s1, a, s2] += prob
                if done:
                    env._terminal_state_mask[s2] = 1.0

    # Sanity check - is the transition matrix valid
    for s1 in env._states:
        for a in env._actions:
            transition_prob = np.sum(env._t_mat[s1, a, :])
            if transition_prob < 1.0:
                warnings.warn(
                    "This environment has inconsistent dynamics - normalizing state-action {}-{}!".format(
                        s1, a
                    )
                )
            env._t_mat[s1, a, :] /= transition_prob

    setattr(EnvClass, "t_mat", property(lambda self: self._t_mat))
    setattr(
        EnvClass,
        "terminal_state_mask",
        property(lambda self: self._terminal_state_mask),
    )

    env._parents, env._children = compute_parents_children(
        env._t_mat, env._terminal_state_mask
    )
    setattr(EnvClass, "parents", property(lambda self: self._parents))
    setattr(EnvClass, "children", property(lambda self: self._children))
    env._gamma = gamma
    setattr(EnvClass, "gamma", property(lambda self: self._gamma))

    env._state_rewards = None
    env._state_action_rewards = None
    env._state_action_state_rewards = None

    # Infer reward structure from transition tensor
    _rs = {s: set() for s in env._states}
    _rsa = {sa: set() for sa in it.product(env._states, env._actions)}
    _rsas = {sas: set() for sas in it.product(env._states, env._actions, env._states)}
    for s1 in env._states:

        if env._terminal_state_mask[s1]:
            # Don't consider transitions from terminal states as they are invalid
            continue

        for a in env._actions:
            for prob, s2, r, done in env.P[s1][a]:
                _rs[s2].add(r)
                _rsa[(s1, a)].add(r)
                _rsas[(s1, a, s2)].add(r)
    _rs = {s: list(rewards) for s, rewards in _rs.items()}
    _rsa = {sa: list(rewards) for sa, rewards in _rsa.items()}
    _rsas = {sas: list(rewards) for sas, rewards in _rsas.items()}
    num_rewards_per_state = [len(rewards) for rewards in _rs.values()]
    num_rewards_per_state_action = [len(rewards) for rewards in _rsa.values()]
    num_rewards_per_state_action_state = [len(rewards) for rewards in _rsas.values()]

    if max(num_rewards_per_state) == 1:
        # This MDP allows for consistent state-only rewards
        env._state_rewards = np.array([_rs[s][0] for s in env._states])
    elif max(num_rewards_per_state_action) == 1:
        # This MDP allows for consistent state-action rewards
        env._state_action_rewards = np.zeros((env.nS, env.nA))
        for s1, a in it.product(env._states, env._actions):
            if env._terminal_state_mask[s1]:
                continue
            env._state_action_rewards[s1, a] = _rsa[(s1, a)][0]
    else:
        # This MDP requires state-action-state rewards
        assert (
            max(num_rewards_per_state_action_state) == 1
        ), "MDP rewards are stochastic and can't be represented by a linear reward function"
        env._state_action_state_rewards = np.zeros((env.nS, env.nA, env.nS))
        for s1, a, s2 in it.product(env._states, env._actions, env._states):
            if env._terminal_state_mask[s1]:
                continue
            env._state_action_state_rewards[s1, a, s2] = _rsas[(s1, a, s2)][0]

    setattr(EnvClass, "state_rewards", property(lambda self: self._state_rewards))
    setattr(
        EnvClass,
        "state_action_rewards",
        property(lambda self: self._state_action_rewards),
    )
    setattr(
        EnvClass,
        "state_action_state_rewards",
        property(lambda self: self._state_action_state_rewards),
    )

    return env
