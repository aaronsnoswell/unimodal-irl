import numpy as np

from numba import jit


def smq_value_iteration(env, beta=0.5, eps=1e-6, verbose=False, max_iter=None):
    """Value iteration to find the SoftMax-optimal state-action value function
    
    This bellman recursion is defined in Section 3 of Apprenticeship Learning about
    Multiple Intentions by Babes-Vroman et al. 2011
    
    http://www.icml-2011.org/papers/478_icmlpaper.pdf
    
    Args:
        env (.envs.explicit.IExplicitEnv) Explicit Gym environment
        
        beta (float): Boltzmann exploration policy temperature parameter
        eps (float): Value convergence tolerance
        verbose (bool): Extra logging
        max_iter (int): If provided, iteration will terminate regardless of convergence
            after this many iterations.
    
    Returns:
        (numpy array): |S|x|A| matrix of state-action values
    """

    rs = env.state_rewards
    if rs is None:
        rs = np.zeros(env.t_mat.shape[0], dtype=np.float)

    rsa = env.state_action_rewards
    if rsa is None:
        rsa = np.zeros(env.t_mat.shape[0:2], dtype=np.float)

    rsas = env.state_action_state_rewards
    if rsas is None:
        rsas = np.zeros(env.t_mat.shape[0:3], dtype=np.float)

    return _nb_smq_value_iteration(
        env.t_mat,
        env.gamma,
        rs,
        rsa,
        rsas,
        beta=beta,
        eps=eps,
        verbose=verbose,
        max_iter=max_iter,
    )


@jit(nopython=True)
def _nb_smq_value_iteration(
    t_mat, gamma, rs, rsa, rsas, beta=0.5, eps=1e-6, verbose=False, max_iter=None
):
    """Value iteration to find the SoftMax-optimal state-action value function
    
    This bellman recursion is defined in Section 3 of Apprenticeship Learning about
    Multiple Intentions by Babes-Vroman et al. 2011
    (http://www.icml-2011.org/papers/478_icmlpaper.pdf).
    
    Essentially, the max over actions from the regular Q-function is replaced with
    an operator that averages over all possible actions, where the weight of each
    Q(s, a) is given by e^{βQ(s, a)} / Σ_{a'} e^{βQ(s, a')}.
    
    Args:
        t_mat (numpy array): |S|x|A|x|S| transition matrix
        gamma (float): Discount factor
        rs (numpy array): |S| State reward vector
        rsa (numpy array): |S|x|A| State-action reward vector
        rsas (numpy array): |S|x|A|x|S| State-action-state reward vector
        
        beta (float): Boltzmann exploration policy temperature parameter
        eps (float): Value convergence tolerance
        verbose (bool): Extra logging
        max_iter (int): If provided, iteration will terminate regardless of convergence
            after this many iterations.
    
    Returns:
        (numpy array): |S|x|A| matrix of state-action values
    """

    q_value_fn = np.zeros((t_mat.shape[0], t_mat.shape[1]))

    _iter = 0
    while True:
        delta = 0

        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):

                # q_weights defines the weight of each Q(s, a) term in the
                # SoftMax operator
                q_weights = np.exp(q_value_fn.copy() * beta)
                norm = np.sum(q_weights, axis=1)
                # Normalize weights to proper probabilities
                for _a in range(q_weights.shape[1]):
                    q_weights[:, _a] = q_weights[:, _a] / norm

                q = q_value_fn[s1, a]
                state_values = np.zeros(t_mat.shape[0])
                for s2 in range(t_mat.shape[2]):
                    state_values[s2] += t_mat[s1, a, s2] * (
                        rs[s1]
                        + rsa[s1, a]
                        + rsas[s1, a, s2]
                        + gamma
                        * (q_value_fn[s2, :].flatten() @ q_weights[s2, :].flatten())
                    )
                q_value_fn[s1, a] = np.sum(state_values)
                delta = max(delta, np.abs(q - q_value_fn[s1, a]))

        if max_iter is not None and _iter >= max_iter:
            if verbose:
                print("Terminating before convergence, # iterations = ", _iter)
                break

        # Check value function convergence
        if delta < eps:
            break
        else:
            if verbose:
                print("Value Iteration #", _iter, " delta=", delta)

        _iter += 1

    return q_value_fn


def main():
    """Main function"""

    # Test functionality
    from explicit_env.envs.explicit_frozen_lake import ExplicitFrozenLakeEnv
    from explicit_env.soln import q_value_iteration

    env = ExplicitFrozenLakeEnv()
    env._gamma = 0.99
    print(env.reward_range)

    q_fn = q_value_iteration(env)
    print(q_fn)

    smq_fn = smq_value_iteration(env, beta=0.5)
    print(smq_fn)


if __name__ == "__main__":
    main()
