import copy
import numpy as np
import warnings

import itertools as it

from numba import jit

from scipy.optimize import minimize

from explicit_env.soln import BoltzmannExplorationPolicy, OptimalPolicy

from explicit_env.soln import value_iteration, q_value_iteration
from unimodal_irl.utils import empirical_feature_expectations, minimize_vgd


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
    rollouts,
    boltzmann_scale=0.5,
    qge="fpi",
    qge_fpi_tol=1e-5,
    qge_max_path_length=None,
    qge_rollouts_per_sa=100,
    nll_only=False,
    verbose=False,
):
    """Compute Negative Log Likelihood (and gradient) for ML-IRL
    
    TODO ajs 29/Oct/2020 Support SoftMax Q function from Babes-Vroman 2011 paper via
        smq_value_iteration()
    
    Args:
        theta_sa (numpy array): Flat |S|x|A| reward parameter weights
        env (IExplicitEnv): Environment
        rollouts (list): List of (s, a) rollouts.
        
        boltzmann_scale (float):
        qge (str): Method of Q-Value Gradient Estimation, one of
            'fpi' - Use fixed-point iteration
            'sim' - Use simulation (sample from optimal policy)
        qge_fpi_tol (float): Convergence threshold for Fixed Point Iteration, only used
            if qge == 'fpi'.
        qge_max_path_length (int): Maximum path length to sample. Only used if qge == 'sim'
            and only needed if the environment is non-episodic.
        qge_rollouts_per_sa (int): Number of rollouts to sample for each (s, a) pair - only
            used if qge == 'sim'.
        nll_only (bool): If true, skip gradient estimation and only compute NLL.
        verbose (bool): Log status information
    """

    if not nll_only:
        assert qge in ["fpi", "sim"], f"Invalid value '{qge}' passed for qge"

    nll_sa._call_count += 1
    if verbose:
        print("Obj#{}".format(nll_sa._call_count))
        print(theta_sa)

    feature_dim = len(env.states) * len(env.actions)

    # Compute Q*, pi* for current reward guess
    env._state_action_rewards = theta_sa.reshape(len(env.states), len(env.actions))
    q = q_value_iteration(env)
    # q = smq_value_iteration(env, boltzmann_scale)
    pi = BoltzmannExplorationPolicy(q, scale=boltzmann_scale)

    if not nll_only:
        if qge == "fpi":
            q_grad = q_grad_fpi(env, theta_sa, tol=qge_fpi_tol)
        elif qge == "sim":
            q_grad = q_grad_simulation(
                env,
                theta_sa,
                rollouts_per_sa=qge_rollouts_per_sa,
                max_rollout_length=qge_max_path_length,
            )
        else:
            raise ValueError(f"Invalid value '{qge}' passed for qge")

    # Sweep demonstrated state-action pairs
    nll = 0
    nll_grad = np.zeros_like(theta_sa)
    for path in rollouts:
        for s, a in path[:-1]:
            ell_theta = pi.prob_for_state_action(s, a)

            # Accumulate negative log likelihood of demonstration data
            nll += -1 * np.log(ell_theta)

            if not nll_only:
                # Estimate gradient of negative log likelihood wrt. parameters
                nll_grad += boltzmann_scale * (
                    np.sum(
                        [
                            pi.prob_for_state_action(s, b) * q_grad[s, b, :]
                            for b in env.actions
                        ]
                    )
                    - q_grad[s, a, :]
                )

    if nll_only:
        return nll
    else:
        return nll, nll_grad


# Static objective function call count
nll_sa._call_count = 0


def q_grad_fpi(env, theta_sa, tol=1e-3):
    """Estimate the Q-gradient with a Fixed Point Iteration
    
    This method uses a Fixed-Point estimate by Neu and Szepesvari 2007.
    
    C.f. https://github.com/Riley16/scot/blob/master/algorithms/max_likelihood_irl.py
    for another attempted implementation that appears to adapt this FPI for state-only
    features.
    
    Args:
        env (IExplicitEnv): Environment
        theta_sa (numpy array): Flat |S|x|A| reward parameter vector
    
    Returns:
        (numpy array): |S|x|A|x(|S|x|A|) Array of partial derivatives δQ(s, a)/dθ
    """

    # Get optimal policy
    env._state_action_rewards = theta_sa.reshape(len(env.states), len(env.actions))
    q = q_value_iteration(env)
    pi = OptimalPolicy(q)

    def phi_sa(s, a):
        ret = np.zeros_like(env.state_action_rewards)
        ret[s, a] = 1.0
        return ret.flatten()

    feature_dim = len(env.states) * len(env.actions)

    # Initialize starting point
    dq_dtheta = np.zeros((len(env.states), len(env.actions), feature_dim))
    for s in env.states:
        for a in env.actions:
            dq_dtheta[s, a, :] = phi_sa(s, a)

    # Apply fixed point iteration
    for iter in it.count():
        # Use full-width backups
        dq_dtheta_old = dq_dtheta.copy()
        dq_dtheta[:, :, :] = 0.0
        for s1 in env.states:
            for a1 in env.actions:
                dq_dtheta[s1, a1, :] = phi_sa(s1, a1)
                for s2 in env.states:
                    for a2 in env.actions:
                        dq_dtheta[s1, a1, :] += (
                            env.gamma
                            * env.t_mat[s1, a1, s2]
                            * pi.prob_for_state_action(s2, a2)
                            * dq_dtheta_old[s2, a2, :]
                        )
                pass

        delta = np.max(np.abs(dq_dtheta_old.flatten() - dq_dtheta.flatten()))

        if delta <= tol:
            break

    return dq_dtheta


def q_grad_simulation(env, theta_sa, rollouts_per_sa=100, max_rollout_length=None):
    """Estimate the Q-gradient with simulation
    
    This method samples many rollouts from the optimal stationary stochastic policy for
    every possible (s, a) pair.
    
    Args:
        env (IExplicitEnv): Environment
        theta_sa (numpy array): Flat |S|x|A| reward parameter vector
        
        rollouts_per_sa (int): Number of rollouts to sample for each (s, a) pair. If
            the environment has deterministic dynamics, it's OK to set this to a small
            number.
        max_rollout_length (int): Maximum rollout length - only needed for non-episodic
            MDPs.
    
    Returns:
        (numpy array): |S|x|A|x(|S|x|A|) Array of partial derivatives δQ(s, a)/dθ
    """

    # Get optimal policy
    env._state_action_rewards = theta_sa.reshape(len(env.states), len(env.actions))
    q = q_value_iteration(env)
    pi = OptimalPolicy(q, stochastic=True)

    feature_dim = len(env.states) * len(env.actions)

    # Calculate expected feature vector under pi for all starting state-action pairs
    dq_dtheta = np.zeros((len(env.states), len(env.actions), feature_dim))
    for s in env.states:
        for a in env.actions:
            # Start with desired state, action
            forced_sa_rollouts = pi.get_rollouts(
                env,
                rollouts_per_sa,
                max_path_length=max_rollout_length,
                start_state=s,
                start_action=a,
            )
            _, phi_sa, _ = empirical_feature_expectations(env, forced_sa_rollouts)
            dq_dtheta[s, a, :] = phi_sa.flatten()

    return dq_dtheta


def bv_maxlikelihood_irl(
    env,
    rollouts,
    boltzmann_scale=0.5,
    max_iterations=None,
    qge="fpi",
    qge_fpi_tol=1e-5,
    qge_max_path_length=None,
    qge_rollouts_per_sa=100,
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
        qge (str): Method of Q-Value Gradient Estimation (ignored if
            grad_twopoint == True). Can be one of
                'fpi' - Use fixed-point iteration
                'sim' - Use simulation (sample from optimal policy)
        qge_fpi_tol (float): Convergence threshold for Fixed Point Iteration, only used
            if qge == 'fpi'.
        qge_max_path_length (int): Maximum path length to sample. Only used if qge == 'sim'
            and only needed if the environment is non-episodic.
        qge_rollouts_per_sa (int): Number of rollouts to sample for each (s, a) pair - only
            used if qge == 'sim'.
        grad_twopoint (bool): If true, use two-point numerical difference gradient
            estimation
        verbose (bool): Print progress information

    Returns:
        (numpy array): State-action reward function
    """

    if not grad_twopoint:
        assert qge in ["fpi", "sim"], f"Invalid value '{qge}' passed for qge"

    _env = copy.deepcopy(env)
    num_states = len(_env.states)
    num_actions = len(_env.actions)

    # Reset objective function call counts
    nll_sa._call_count = 0

    # Use scipy minimization procedures
    min_fn = minimize
    # min_fn = minimize_vgd

    # Estimate gradient from two-point NLL numerical difference?
    # Seems to help with convergence for some problems
    if grad_twopoint:
        jac = "2-point"
        nll_only = True
    else:
        jac = True
        nll_only = False
    opt_method = "L-BFGS-B"
    # opt_method = "CG"

    if verbose:
        print("Optimizing state-action rewards")

    options = dict(disp=True)
    if max_iterations is not None:
        options["maxiter"] = max_iterations
    # options["step_size"] = 1e-5

    res = min_fn(
        nll_sa,
        # np.zeros(num_states * num_actions) + np.mean(env.reward_range),
        np.random.uniform(
            env.reward_range[0], env.reward_range[1], num_states * num_actions
        ),
        args=(
            env,
            rollouts,
            boltzmann_scale,
            qge,
            qge_fpi_tol,
            qge_max_path_length,
            qge_rollouts_per_sa,
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
