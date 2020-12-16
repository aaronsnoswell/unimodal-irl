import copy
import numpy as np
import warnings

import itertools as it

from numba import jit

from explicit_env.soln import BoltzmannExplorationPolicy

from mdp_extras import Linear, q_vi, q_grad_fpi


@jit(nopython=True)
def nb_smq_value_iteration(
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


def bv_maxlikelihood_irl(
    x, xtr, phi, rollouts, weights=None, boltzmann_scale=0.5, qge_tol=1e-3, nll_only=False
):
    """Compute the average rollout Negative Log Likelihood (and gradient) for ML-IRL
    
    This method is biased to prefer shorter paths through any MDP.
    
    TODO ajs 29/Oct/2020 Support SoftMax Q function from Babes-Vroman 2011 paper via
        nb_smq_value_iteration()
    
    Args:
        x (numpy array): Current reward function parameter vector estimate
        xtr (mdp_extras.DiscreteExplicitExtras): Extras object for the MDP being
            optimized
        phi (mdp_extras.FeatureFunction): Feature function to use with linear reward
            parameters. We require len(phi) == len(x).
        rollouts (list): List of (s, a) rollouts.
        
        weights (numpy array): Optional path weights for weighted IRL problems
        boltzmann_scale (float): Optimality parameter for Boltzmann policy. Babes-Vroman
            use 0.5. Values closer to 1.0 cause slower convergence, but values closer to
            0 model the demonstrations as being non-expert. Empirically I find 0.2 works
            well.
        qge_tol (float): Tolerance for q-gradient estimation.
        nll_only (bool): If true, only return NLL
    """

    if weights is None:
        weights = np.ones(len(rollouts)) / len(rollouts)

    # Compute Q*, pi* for current reward guess
    reward = Linear(x)
    q_star = q_vi(xtr, phi, reward)

    # To use the soft Q function from Babes-Vroman's paper, uncomment below
    # q_star = nb_smq_value_iteration(
    #     xtr.t_mat, xtr.gamma, *reward.structured(xtr, phi), boltzmann_scale
    # )
    pi = BoltzmannExplorationPolicy(q_star, scale=boltzmann_scale)

    if not nll_only:
        # Get Q* gradient for current reward parameters
        dq_dtheta = q_grad_fpi(reward.theta, xtr, phi, tol=qge_tol)

    # Sweep demonstrated state-action pairs
    nll = 0
    nll_grad = np.zeros_like(x)
    num_sa_samples = 0
    for path, weight in zip(rollouts, weights):
        for s, a in path[:-1]:
            num_sa_samples += 1
            ell_theta = pi.prob_for_state_action(s, a)

            # Accumulate negative log likelihood of demonstration data
            nll += -1 * weight * np.log(ell_theta)

            if not nll_only:
                # XXX b here is an auxillary state - dq_dtheta indexerror
                expected_action_grad = np.sum(
                    [
                        pi.prob_for_state_action(s, b) * dq_dtheta[s, b, :]
                        for b in xtr.unpadded.actions
                    ],
                    axis=0,
                )
                dl_dtheta = boltzmann_scale * (
                    expected_action_grad - dq_dtheta[s, a, :]
                )
                nll_grad += weight * dl_dtheta

    # Convert NLL and gradient to average, not sum
    # This makes for consistent magnitude values regardless of dataset size
    nll /= len(rollouts)
    nll_grad /= len(rollouts)

    if nll_only:
        return nll
    else:
        return nll, nll_grad


def main():
    """Main function"""
    pass


if __name__ == "__main__":
    main()
