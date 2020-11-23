"""Various utility methods related to IRL algorithms"""

import gym
import copy
import warnings
import numpy as np

from explicit_env.envs.utils import compute_parents_children

from mdp_extras import DiscreteExplicitExtras, Indicator, Disjoint, Linear


def padding_trick(xtr, phi, r, rollouts=None, max_length=None):
    """Apply padding trick, adding an auxiliary state and action to an MDP
    
    We gain a O(|S|) space and time efficiency improvement with our MaxEnt IRL algorithm
    for MDPs with terminal states by transforming them to have no terminal states. This
    is done by adding a dummy state and action that pad any trajectories out to a fixed
    upper length.
    
    Args:
        xtr (DiscreteExplicitExtras): Extras object
        phi (Indicator): Indicator/Disjoint feature function
        r (Linear): Linear reward function
        
        rollouts (list): List of [(s, a), (s, a), ..., (s, None)] rollouts to pad
        max_length (int): Optional maximum length to pad to, otherwise paths are padded
            to match the length of the longest path
        
    Returns:
        (DiscreteExplicitExtras): Extras object, padded with auxiliary state and action
        (Indicator): Indicator feature function, padded with auxiliary state and action
        (Linear): Linear reward function, padded with auxiliary state and action
        
        (list): List of rollouts, padded to max_length. Only returned if rollouts is not
            None
    """

    t_mat = np.pad(xtr.t_mat, (0, 1), mode="constant")
    s_aux = t_mat.shape[0] - 1
    a_aux = t_mat.shape[1] - 1

    p0s = np.pad(xtr.p0s, (0, 1), mode="constant")
    states = np.arange(t_mat.shape[0])
    actions = np.arange(t_mat.shape[1])

    # Auxiliary state is absorbing
    t_mat[-1, -1, -1] = 1

    # Terminal states are no longer absorbing
    for terminal_state in np.argwhere(xtr.terminal_state_mask):
        t_mat[terminal_state, :, terminal_state] = 0
    terminal_state_mask = np.zeros(t_mat.shape[0])

    # Auxiliary state reachable anywhere if auxiliary action is taken
    t_mat[:, -1, -1] = 1

    xtr2 = DiscreteExplicitExtras(
        states, actions, p0s, t_mat, terminal_state_mask, xtr.gamma
    )

    # Auxiliary state, action don't modify rewards
    if isinstance(phi, Indicator):
        # Pad an indicator feature function and linear reward function
        rs, rsa, rsas = r.structured(xtr, phi)
        rs = np.pad(rs, (0, 1), mode="constant")
        rs[-1] = 0

        rsa = np.pad(rsa, (0, 1), mode="constant")
        rsa[:, -1] = 0

        rsas = np.pad(rsas, (0, 1), mode="constant")
        rsas[:, 0:-1, -1] = -np.inf  # Illegal transition
        rsas[:, -1, -1] = 0

        if phi.type == Indicator.Type.OBSERVATION:
            r2 = Linear(rs.flatten())
        elif phi.type == Indicator.Type.OBSERVATION_ACTION:
            r2 = Linear(rsa.flatten())
        elif phi.type == Indicator.Type.OBSERVATION_ACTION_OBSERVATION:
            r2 = Linear(rsas.flatten())
        else:
            raise ValueError

        phi2 = Indicator(phi.type, xtr2)

    elif isinstance(phi, Disjoint):
        # Pad a disjoint feature function and linear reward function

        phi2 = phi
        r2 = r
    else:
        raise ValueError

    if rollouts is None:
        return xtr2, phi2, r2
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

        # Finally, pad the trajectories out to the maximum length
        rollouts2 = []
        for rollout in rollouts:
            rollout2 = rollout.copy()
            if len(rollout2) < max_length:
                rollout2[-1] = (rollout2[-1][0], a_aux)
                while len(rollout2) != max_length - 1:
                    rollout2.append((s_aux, a_aux))
                rollout2.append((s_aux, None))
            rollouts2.append(rollout2)

        return xtr2, phi2, r2, rollouts2
