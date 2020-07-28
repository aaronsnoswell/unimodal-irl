"""Various utility methods related to Gym environments"""

import copy
import numpy as np

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
