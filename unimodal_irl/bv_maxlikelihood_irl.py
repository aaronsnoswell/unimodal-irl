import numpy as np

from numba import jit

from scipy.optimize import minimize

from explicit_env.soln import BoltzmannExplorationPolicy

from utils import empirical_feature_expectations


def smq_value_iteration(env, beta=0.5, eps=1e-6, verbose=False, max_iter=None):
    """Value iteration to find the SoftMax-optimal state-action value function
    
    This bellman recursion is defined in Section 3 of Apprenticeship Learning about
    Multiple Intentions by Babes-Vroman et al. 2011
    
    http://www.icml-2011.org/papers/478_icmlpaper.pdf
    
    Args:
        env (.envs.explicit.IExplicitEnv) Explicit Gym environment
        
        beta (float): Boltzmann exploration policy scale parameter
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
        
        beta (float): Boltzmann exploration policy scale parameter
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


def nll_sa(
    theta_sa,
    env,
    boltzmann_scale,
    rollouts,
    num_rollouts_phibar_calc,
    phibar_calc_max_rollout_length=None,
    nll_only=False,
    rescale_grad=False,
    verbose=False,
):
    nll_sa._call_count += 1
    if verbose:
        print("Obj#{}".format(nll_sa._call_count))
        print(theta_sa)
    env._state_action_rewards = theta_sa.reshape((len(env.states), len(env.actions)))
    with np.errstate(over="raise"):

        # Find Q_theta using smq_value_iteration
        smq_theta = smq_value_iteration(env, beta=boltzmann_scale)

        # Form boltzmann policy from Q_theta
        # This is used to compute likelihood of (s,a) pairs ell_theta[s,a]
        pi_b = BoltzmannExplorationPolicy(smq_theta, scale=boltzmann_scale)

        # Compute likelihoods of state-action pairs
        ell_sa = np.zeros((len(env.states), len(env.actions)))
        for s in env.states:
            ell_sa[s, :] = pi_b.prob_for_state(s)

        # Compute expected feature vector under policy
        # _, Phi_sa, _ = empirical_feature_expectations(
        #     env, pi_b.get_rollouts(env, num_rollouts_phibar_calc, max_path_length=30)
        # )
        # Get a crapload of rollouts
        _rollouts = [
            pi_b.get_rollouts(
                env,
                num_rollouts_phibar_calc,
                max_path_length=phibar_calc_max_rollout_length,
                first_state=_s0,
                first_action=_a0,
            )
            for _s0 in env.states
            for _a0 in env.actions
        ]
        Phi_s = np.zeros((len(env.states)))
        Phi_sa = np.zeros((len(env.states), len(env.actions)))
        Phi_sas = np.zeros((len(env.states), len(env.actions), len(env.states)))
        for _r in _rollouts:
            for t, (s, _) in enumerate(_r):
                pass
            for t, (s, a) in enumerate(_r[:-1]):
                pass
            for t, ((s1, a), (s2, _)) in enumerate(zip(_r[:-1], _r[1:])):
                pass

        # Compute gradient of likelihoods of pairs using Phi_sa
        # This step is an approximation by Neu and Szepesvari, 2007
        duh_ell_duh_theta = np.zeros((len(env.states), len(env.actions)))
        for s in env.states:
            for a in env.actions:
                duh_ell_duh_theta[s, a] = (
                    boltzmann_scale
                    * ell_sa[s, a]
                    * (Phi_sa[s, a] - ell_sa[s, :] @ Phi_sa[s, :])
                )

        # Accumulate log likelihood and log likelihood gradient contributions
        nll = 0
        grad = np.zeros((len(env.states), len(env.actions)))
        for rollout in rollouts:
            for s, a in rollout[:-1]:
                nll -= np.log(ell_sa[s, a])
                grad[s, a] += 1.0 / ell_sa[s, a] * duh_ell_duh_theta[s, a]
        grad = grad.flatten()

        if verbose:
            print(nll)
            print(grad)

        if nll_only:
            return nll
        else:
            if rescale_grad:
                if verbose:
                    print(
                        f"Re-scaling gradient by a factor of 1/{np.linalg.norm(grad)}"
                    )
                grad /= np.linalg.norm(grad)
            return nll, grad


# Static objective function call count
nll_sa._call_count = 0


def bv_maxlikelihood_irl(
    env, rollouts, *, boltzmann_scale=0.5, num_rollouts=1000, verbose=True,
):
    """Find Reward function using ML-IRL
    
    See: http://www.icml-2011.org/papers/478_icmlpaper.pdf
    Also: https://arxiv.org/pdf/1202.1558.pdf
    Also: https://arxiv.org/ftp/arxiv/papers/1206/1206.5264.pdf
    Also: https://www.overleaf.com/project/5edf22df9b1f270001507c4d
    
    Example implementation (this implementation gets the Softmax Q-function from the
    Babes-Vroman paper wrong, but appears to get other details right like the gradient
    estimation)
    https://github.com/Riley16/scot/blob/master/algorithms/max_likelihood_irl.py
    
    Args:
        env (explicit_env.IExplicitEnv): Reward-less environment to solve for
        rollouts (list): List of demonstration state-action rollouts
        
        boltzmann_scale (float): Boltzmann policy scale parameter. Values -> infinity
            leads to optimal policy, but no exploration.
        num_rollouts (int): Number of rollouts to use for empirical calculation of the
            Boltzman policy feature expectation(s)
        verbose (bool): Print progress information
    
    Returns:
        None
    """

    num_states = len(env.states)
    num_actions = len(env.actions)

    # Use scipy minimization procedures
    min_fn = minimize

    # Estimate gradient from two-point NLL numerical difference?
    # Seems to help with convergence for some problems
    grad_twopoint = True
    if grad_twopoint:
        jac = "2-point"
        nll_only = True
    else:
        jac = True
        nll_only = False
    rescale_grad = False
    opt_method = "L-BFGS-B"

    # Reset objective function call counts
    nll_sa._call_count = 0

    if verbose:
        print("Optimizing state-action rewards")
    res = min_fn(
        nll_sa,
        np.zeros(num_states * num_actions) + np.mean(env.reward_range),
        args=(
            env,
            boltzmann_scale,
            rollouts,
            num_rollouts,
            nll_only,
            rescale_grad,
            True
            # False,  # verbose,
        ),
        method=opt_method,
        jac=jac,
        bounds=tuple(env.reward_range for _ in range(num_states * num_actions)),
        options=dict(disp=True),
    )

    print("Done")


# # Algorithm from https://github.com/Riley16/scot/blob/master/algorithms/max_likelihood_irl.py for
# # for computing the expected feature counts. Seems to be total Bullshit.
# def get_feature_counts(env, pi, tol=1e-6):
#     """Get expected linear feature counts under a policy
#
#     Args:
#
#     """
#
#     # s-s' probability under policy and dynamics
#     pol_trans = np.zeros((len(env.states), len(env.states)))
#     for s1 in env.states:
#         for a in env.actions:
#             pol_trans[s1, :] += pi.prob_for_state_action(s1, a) * env.t_mat[s1, a, :]
#
#     with np.errstate(all="raise"):
#         mu_s = np.zeros((len(env.states), len(env.states)))
#         eps = np.inf
#         while eps > tol:
#             mu_s_new = np.zeros_like(mu_s)
#             for s1 in env.states:
#                 phi_s = np.zeros_like(env.states)
#                 phi_s[s1] = 1.0
#                 mu_s_new[s1, :] = phi_s + env.gamma * np.sum(
#                     [
#                         pol_trans[s1, s2]
#                         * mu_s[s2, :]
#                         * (not env.terminal_state_mask[s1])
#                         for s2 in env.states
#                     ]
#                 )
#             eps = np.max(np.abs(mu_s_new - mu_s))
#             print(eps)
#             mu_s = mu_s_new
#
#     print(mu_s)


def main():
    """Main function"""

    # Test functionality
    from explicit_env.envs.explicit_frozen_lake import ExplicitFrozenLakeEnv
    from explicit_env.envs.explicit_nchain import ExplicitNChainEnv
    from explicit_env.soln import q_value_iteration, OptimalPolicy

    env = ExplicitFrozenLakeEnv()
    env._gamma = 0.1
    # env = ExplicitNChainEnv()
    # env._gamma = 0.99
    pi = OptimalPolicy(q_value_iteration(env))

    # # Get rollouts
    # num_rollouts = 20
    # rollouts = pi.get_rollouts(
    #     env, num_rollouts, max_path_length=30
    # )
    # reward_weights = bv_maxlikelihood_irl(env, rollouts)
    # print(reward_weights)


if __name__ == "__main__":
    main()
