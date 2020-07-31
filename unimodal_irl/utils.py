"""Various utility methods related to IRL algorithms"""

import numpy as np
import itertools as it


def get_rollouts(env, policy, num_rollouts, *, max_episode_length=None, verbose=False):
    """Get rollouts of a policy in an environment
    
    Args:
        env (gym.Env): Environment to use
        policy (object): Policy object with a .predict() method matching the
            stable-baselines policy API.
        num_rollouts: Number of rollouts to collect
        
        max_episode_length (int): If provided, stop trajectories at this length
        verbose (bool): If true, print extra logging info
        
    Returns:
        (list): List of [(s, a), (s, a), ..., (s, None)] trajectories
    """

    rollouts = []
    for episode in it.count():

        # Prepare one trajectory
        rollout = []

        # Reset environment
        s = env.reset()
        for timestep in it.count():

            # Sample action from policy
            a, _ = policy.predict(s)

            # Add state, action to trajectory
            rollout.append((s, a))

            # Step environment
            s, r, done, _ = env.step(a)

            if done:
                break

            if max_episode_length is not None:
                if timestep == max_episode_length - 2:
                    if verbose:
                        print("Stopping after reaching maximum episode length")
                    break

        rollout.append((s, None))
        rollouts.append(rollout)

        if episode == num_rollouts - 1:
            break

    return rollouts


def empirical_feature_expectations(env, rollouts):
    """Find empirical discounted feature expectations
    
    Args:
        env (unimodal_irl.envs.explicit.IExplicitEnv): Environment defining dynamics,
            reward(s) and discount factor
        rollouts (list): List of [(s, a), (s, a), ..., (s, None)] trajectories
    
    Returns:
        (numpy array): |S| array of state marginals
        (numpy array): |S|x|A| array of state-action marginals
        (numpy array): |S|x|A|x|S| array of state-action-state marginals
    """

    # Find discounted feature expectations
    phibar_s = np.zeros(env.t_mat.shape[0])
    phibar_sa = np.zeros(env.t_mat.shape[0 : 1 + 1])
    phibar_sas = np.zeros(env.t_mat.shape)
    for r in rollouts:

        if env.state_rewards is not None:
            for t, (s1, _) in enumerate(r):
                phibar_s[s1] += (env.gamma ** t) * 1

        if env.state_action_rewards is not None:
            for t, (s1, a) in enumerate(r[:-1]):
                phibar_sa[s1, a] += (env.gamma ** t) * 1

        if env.state_action_state_rewards is not None:
            for t, (s1, a) in enumerate(r[:-1]):
                s2 = r[t + 1][0]
                phibar_sas[s1, a, s2] += (env.gamma ** t) * 1

    return phibar_s, phibar_sa, phibar_sas
