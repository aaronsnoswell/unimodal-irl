import copy
import numpy as np


def compute_parents_children(t, terminal_state_mask):
    """Compute parent and child dictionaries"""
    parents = {s: [] for s in range(t.shape[0])}
    children = copy.deepcopy(parents)
    for s2 in range(t.shape[0]):
        for s1, a in np.argwhere(t[:, :, s2] > 0):
            if not terminal_state_mask[s1]:
                parents[s2].append(tuple((s1, a)))
                children[s1].append(tuple((a, s2)))

    return parents, children


def pad_mdp(env):
    """Pads an MDP, adding a dummy state and action"""

    # Add an extra state and action to the dynamics
    t2 = np.pad(env.t, (0, 1), mode="constant")
    p0s2 = np.pad(env.p0s, (0, 1), mode="constant")
    terminal_state_mask2 = np.pad(env.terminal_state_mask, (0, 1), mode="constant")

    state_rewards2 = None
    state_action_rewards2 = None
    state_action_state_rewards2 = None

    # Dummy state is absorbing
    t2[-1, -1, -1] = 1

    # Terminal states are no longer absorbing
    for terminal_state in np.argwhere(env.terminal_state_mask):
        t2[terminal_state, :, terminal_state] = 0
    terminal_state_mask2 = np.zeros(t2.shape[0])

    # Dummy state reachable anywhere if dummy action is taken
    t2[:, -1, -1] = 1

    # Dummy state doesn't modify rewards
    if hasattr(env, "state_rewards"):
        state_rewards2 = np.pad(env.state_rewards, (0, 1), mode="constant")
        state_rewards2[-1] = 0
    if hasattr(env, "state_action_rewards"):
        state_action_rewards2 = np.pad(
            env.state_action_rewards, (0, 1), mode="constant"
        )
        state_action_rewards2[:, -1] = 0
    if hasattr(env, "state_action_state_rewards"):
        state_action_state_rewards2 = np.pad(
            env.state_action_state_rewards, (0, 1), mode="constant"
        )
        state_action_state_rewards2[:, 0:-1, -1] = -np.inf  # Illegal transition
        state_action_state_rewards2[:, -1, -1] = 0

    # Overwrite environment properties
    env.t = t2
    env.p0s = p0s2
    env.terminal_state_mask = terminal_state_mask2
    env.states = np.arange(t2.shape[0])
    env.actions = np.arange(t2.shape[1])

    if hasattr(env, "state_rewards"):
        env.state_rewards = state_rewards2
    if hasattr(env, "state_action_rewards"):
        env.state_action_rewards = state_action_rewards2
    if hasattr(env, "state_action_state_rewards"):
        env.state_action_state_rewards = state_action_state_rewards2

    return env
