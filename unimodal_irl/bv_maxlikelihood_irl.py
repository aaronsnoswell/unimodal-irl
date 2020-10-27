import copy
import numpy as np
import warnings

import itertools as it

from numba import jit

from scipy.optimize import minimize

from explicit_env.soln import BoltzmannExplorationPolicy

from explicit_env.soln import q_value_iteration
from unimodal_irl.utils import empirical_feature_expectations


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
    max_path_length,
    num_rollouts_per_sa_pair,
    nll_only=False,
    verbose=False,
):
    nll_sa._call_count += 1
    if verbose:
        print("Obj#{}".format(nll_sa._call_count))
        print(theta_sa)

    feature_dim = len(env.states) * len(env.actions)

    # Compute Q*, pi* for current reward guess
    env._state_action_rewards = theta_sa.reshape(len(env.states), len(env.actions))
    q = q_value_iteration(env)
    pi = BoltzmannExplorationPolicy(q, scale=boltzmann_scale)

    if not nll_only:
        # Calculate expected feature vector under pi for all starting state-actions
        phi_pi_sa = np.zeros((len(env.states), len(env.actions), feature_dim))
        for s in env.states:
            # Force desired starting state
            env._p0s = np.zeros_like(env.p0s)
            env._p0s[s] = 1
            for a in env.actions:
                # Start in state s, action a
                env.reset()
                forced_sa_rollouts = pi.get_rollouts(
                    env,
                    num_rollouts_per_sa_pair,
                    max_path_length=max_path_length,
                    start_action=a,
                )
                _, phi_sa, _ = empirical_feature_expectations(env, forced_sa_rollouts)
                phi_pi_sa[s, a, :] = phi_sa.flatten()

    # Sweep demonstrated state-action pairs
    log_likelihood = 0
    nabla = np.zeros_like(theta_sa)
    for path in rollouts:
        for s, a in path[:-1]:
            ell_theta = pi.prob_for_state_action(s, a)

            # Accumulate log likelihood of demonstration data
            log_likelihood += np.log(ell_theta)

            if not nll_only:
                # Estimate gradient of log likelihood wrt. parameters
                nabla += boltzmann_scale * (
                    phi_pi_sa[s, a, :]
                    - np.sum(
                        [
                            pi.prob_for_state_action(s, b) * phi_pi_sa[s, b, :]
                            for b in env.actions
                        ]
                    )
                )

    nll = -1 * log_likelihood
    nll_grad = -1 * nabla

    if nll_only:
        return nll
    else:
        return nll, nll_grad


# Static objective function call count
nll_sa._call_count = 0


def bv_maxlikelihood_irl_take2(
    env,
    rollouts,
    boltzmann_scale=0.5,
    max_iterations=None,
    max_path_length=None,
    num_rollouts_per_sa_pair=10,
    grad_twopoint=True,
    verbose=False,
):
    """Find Reward function using Max Likelihood IRL

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
        max_iterations (int): Maximum number of optimization steps.
        max_path_length (int): Max path length to use when sampling rollouts for
            estimation of likelihood gradient. Only needed for continuing (non episodic)
            MDPs. Also only needed if grad_twopoint=False.
        num_rollouts_per_sa_pair (int): Number of rollouts to sample for each (s, a)
            pair for estimation of likelihood gradient. If environment is deterministic,
            this can be set to 1. Only needed if grad_twopoint=False.
        grad_twopoint (bool): If true, use two-point numerical difference gradient
            estimation rather than fixed-point gradient estimation by Neu and
            Szepesvari 2007.
            XXX ajs 27/Oct/2020 grad_twopoint=False doesn't seem to work atm.
        verbose (bool): Print progress information

    Returns:
        (numpy array): State-action reward function
    """

    _env = copy.deepcopy(env)
    num_states = len(_env.states)
    num_actions = len(_env.actions)

    # Reset objective function call counts
    nll_sa._call_count = 0

    # Use scipy minimization procedures
    min_fn = minimize

    # Estimate gradient from two-point NLL numerical difference?
    # Seems to help with convergence for some problems
    if grad_twopoint:
        jac = "2-point"
        nll_only = True
    else:
        jac = True
        nll_only = False
    opt_method = "L-BFGS-B"

    if verbose:
        print("Optimizing state-action rewards")
    options = dict(disp=True)
    if max_iterations is not None:
        options["maxiter"] = max_iterations
    res = minimize(
        nll_sa,
        np.zeros(num_states * num_actions) + np.mean(env.reward_range),
        args=(
            env,
            boltzmann_scale,
            rollouts,
            max_path_length,
            num_rollouts_per_sa_pair,
            nll_only,
            verbose,
        ),
        method=opt_method,
        jac=jac,
        bounds=tuple(env.reward_range for _ in range(num_states * num_actions)),
        options=options,
    )

    if not res.success:
        warnings.warn("Minimization did not succeed")
        print(res)

    theta_sa = res.x.reshape((num_states, num_actions))
    if verbose:
        print(res)
        print(
            "Completed optimization after {} iterations, NLL = {}".format(
                res.nit, res.fun
            )
        )
        print("theta_sa = {}".format(theta_sa))

    return theta_sa


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
    from explicit_env.envs.explicit_nchain import ExplicitNChainEnv
    from explicit_env.soln import q_value_iteration, OptimalPolicy
    from unimodal_irl.metrics import ile_evd

    env = ExplicitNChainEnv()
    env._gamma = 0.99
    rsa = env.state_action_rewards
    pi = OptimalPolicy(q_value_iteration(env))
    max_path_length = 30

    # Get rollouts
    # TODO ajs 27/Oct/2020 Using Neu gradient estimation doesn't seem to work
    num_rollouts = 200
    rollouts = pi.get_rollouts(env, num_rollouts, max_path_length=max_path_length)
    rsa_learned = bv_maxlikelihood_irl_take2(
        env,
        rollouts,
        max_path_length=max_path_length,
        num_rollouts_per_sa_pair=10,
        grad_twopoint=True,
        verbose=False,
    )

    print("------------------------------------------")
    print("GT Reward R(s, a) = ")
    print(rsa)

    print()
    print("Learned Reward R(s, a) = ")
    print(rsa_learned)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, sharey=True, gridspec_kw=dict(wspace=0.01))
    plt.set_cmap("Greys_r")

    plt.sca(axes[0])
    plt.imshow(
        rsa, vmin=env.reward_range[0], vmax=env.reward_range[1],
    )
    plt.title(r"$R_{GT}(s, a)$")
    plt.ylabel("States s")
    plt.xlabel("Actions a")

    plt.sca(axes[1])
    plt.imshow(rsa_learned, vmin=env.reward_range[0], vmax=env.reward_range[1])
    plt.title(r"$R_{L}(s, a)$")
    plt.xlabel("Actions a")

    plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.close()

    # UP TO HERE TODO ajs 27/Oct/2020
    # XXX In this case, we require an optimal stochastic policy with q_precision set
    # (and stochastic policy evaluation) to get a realistic ILE, EVD. Otherwise the
    # learned policy defaults to always going right (the optimal policy) due to a float
    # rounding error, and choosing the 0th action amongst many equally good actions.
    # Not sure how to handle these settings elegantly when calling ile_evd()
    # Need to decide on a modified API for ile_evd()
    env_irl = copy.deepcopy(env)
    env_irl._state_action_rewards = rsa_learned
    ile, evd = ile_evd(env, env_irl, verbose=True)

    print(f"ILE = {ile}")
    print(f"EVD = {evd}")

    print("Done")


if __name__ == "__main__":
    main()
