"""Various utility methods related to Gym environments"""

import copy
import warnings
import numpy as np
import itertools as it

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


def pad_terminal_mdp(env, *, rollouts=None, max_length=None):
    """Pads a terminal MDP, adding a dummy state and action
    
    We gain a O(|S|) space and time efficiency improvement with our MaxEnt IRL algorithm
    for MDPs with terminal states by transforming them to have no terminal states. This
    is done by adding a dummy state and action that pad any trajectories out to a fixed
    upper length.
    
    Args:
        env (.explicit.IExplicitEnv) Explicit MDP environment
        
        rollouts (list): List of [(s, a), (s, a), ..., (s, None)] rollouts to pad
        max_length (int): Optional maximum length to pad to, otherwise paths are padded
            to match the length of the longest path
    
    Returns:
        (.explicit.IExplicitEnv) Explicit MDP environment, padded with a dummy
            state and action so that it has no terminal states.
        (list): List of rollouts, padded to max_length
    """

    env = copy.deepcopy(env)

    # Add an extra state and action to the dynamics
    t_mat2 = np.pad(env.t_mat, (0, 1), mode="constant")
    p0s2 = np.pad(env.p0s, (0, 1), mode="constant")
    terminal_state_mask2 = np.pad(env.terminal_state_mask, (0, 1), mode="constant")
    states2 = np.arange(t_mat2.shape[0])
    actions2 = np.arange(t_mat2.shape[1])

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
    env._states = states2
    env._actions = actions2
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

    # Finally, pad the trajectories
    if rollouts is None:
        return env
    else:
        # Measure the length of the rollouts
        r_len = [len(r) for r in rollouts]
        if max_length is None:
            max_length = max(r_len)
        elif max_length < max(r_len):
            warnings.warn(
                f"Provided max length ({max_length}) is < maximum path length ({max(r_len)}), using maximum path length instead"
            )
            max_length = max(r_len)

        _rollouts = []
        dummy_state = t_mat2.shape[0] - 1
        dummy_action = t_mat2.shape[1] - 1
        for r in rollouts:
            _r = r.copy()
            if len(_r) < max_length:
                s, _ = _r[-1]
                _r[-1] = (s, dummy_action)
                while len(_r) != max_length - 1:
                    _r.append((dummy_state, dummy_action))
                _r.append((dummy_state, None))
            _rollouts.append(_r)
        return env, _rollouts


def discrete2explicit(env, *, gamma=1.0):
    """Adds IExplicitEnv protected properties to a DiscreteEnv instance
    
    Args:
        env (gym.envs.toy_text.discrete.DiscreteEnv): Environment to update
        
        gamma (float): Discount factor to assign
    """

    env._states = np.arange(env.nS)
    env._actions = np.arange(env.nA)
    env._p0s = np.array(env.isd)

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

    env._parents, env._children = compute_parents_children(
        env._t_mat, env._terminal_state_mask
    )
    env._gamma = gamma

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
        # print("MDP is consistent with state rewards")
        env._state_rewards = np.zeros(env.nS)
        for s in env._states:
            if len(_rs[s]) == 1:
                env._state_rewards[s] = _rs[s][0]

    elif max(num_rewards_per_state_action) == 1:

        # This MDP allows for consistent state-action rewards
        # print("MDP is consistent with state-action rewards")
        env._state_action_rewards = np.zeros((env.nS, env.nA))
        for s1, a in it.product(env._states, env._actions):
            if env._terminal_state_mask[s1]:
                continue
            if len(_rsa[(s1, a)]) == 1:
                env._state_action_rewards[s1, a] = _rsa[(s1, a)][0]

    else:

        # This MDP requires state-action-state rewards
        assert (
            max(num_rewards_per_state_action_state) == 1
        ), "MDP rewards are stochastic and can't be represented by a linear reward function"
        # print("MDP is consistent with state-action-state rewards")
        env._state_action_state_rewards = np.zeros((env.nS, env.nA, env.nS))
        for s1, a, s2 in it.product(env._states, env._actions, env._states):
            if env._terminal_state_mask[s1]:
                continue
            if len(_rsas[(s1, a, s2)]) == 1:
                env._state_action_state_rewards[s1, a, s2] = _rsas[(s1, a, s2)][0]
