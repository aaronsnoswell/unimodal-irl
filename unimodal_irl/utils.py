"""Various utility methods related to IRL algorithms"""

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

            if timestep == max_episode_length - 2:
                if verbose:
                    print("Stopping after reaching maximum episode length")
                break

        rollout.append((s, None))
        rollouts.append(rollout)

        if episode == num_rollouts:
            break

    return rollouts
