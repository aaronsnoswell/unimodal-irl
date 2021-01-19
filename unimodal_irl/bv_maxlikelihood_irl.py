import copy
import numpy as np
import warnings

import itertools as it

from numba import jit

from mdp_extras import Linear, q_vi, q_grad_fpi, BoltzmannExplorationPolicy


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
    x,
    xtr,
    phi,
    rollouts,
    weights=None,
    boltzmann_scale=0.5,
    qge_tol=1e-3,
    nll_only=False,
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
                expected_action_grad = np.sum(
                    [
                        pi.prob_for_state_action(s, b) * dq_dtheta[s, b, :]
                        for b in xtr.actions
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


def maxlikelihood_ml_path(
    xtr, phi, reward, start, goal, max_path_length, boltzmann_scale=0.5
):
    """Find the ML path from s1 to sg under a MaxLikelihood model
    
    If transitions can inccur +ve rewards the returned paths may contain loops
    
    NB ajs 14/Jan/2020 The log likelihood of the path that we compute internally
        is fine for doing viterbi ML path inference, but it's not the actual path
        log likelihood - it's not normalized, and the gamma time offset
        is incorrect (depending on what start time the Viterbi alg picks).
    
    Args:
        xtr (DiscreteExplicitExtras): MDP Extras object
        phi (FeatureFunction): MDP Featrure function
        reward (Linear): Linear reward function
        start (int): Starting state
        goal (int): End state
        max_path_length (int): Maximum allowable path length to search
        
        boltzmann_scale (float): Boltzmann scale parameter
    
    Returns:
        (list): Maximum Likelihood path from start to goal under the given MaxEnt reward
            function, or None if no path is possible
    """

    q_star = q_vi(xtr, phi, reward)

    # Initialize an SxA LL Viterbi trellis
    sa_lls = np.zeros((len(xtr.states), len(xtr.actions), max_path_length)) - np.inf
    for a in xtr.actions:
        sa_lls[goal, :, :] = boltzmann_scale * q_star[goal, a]

    # Supress divide by zero - we take logs of many zeroes here
    with np.errstate(divide="ignore"):

        # Walk backward to propagate the maximum LL
        for t in range(max_path_length - 2, -1, -1):

            # Max-Reduce over actions to compute state LLs
            # (it's a max because we get to choose our actions)
            s_lls = np.max(sa_lls, axis=1)

            # Sweep end states
            for s2 in xtr.states:

                if np.isneginf(s_lls[s2, t + 1]):
                    # Skip this state - it hasn't been reached by probability messages yet
                    continue

                # Sweep actions
                for a in xtr.actions:

                    # Sweep starting states
                    for s1 in xtr.states:

                        if xtr.terminal_state_mask[s1]:
                            # We can't step forward from terminal states - skip this one
                            continue

                        transition_ll = boltzmann_scale * q_star[s1, a] + np.log(
                            xtr.t_mat[s1, a, s2]
                        )

                        if np.isneginf(transition_ll):
                            # This transition is impossible - skip
                            continue

                        # Store the max because we're after the maximum likelihood path
                        sa_lls[s1, a, t] = max(
                            sa_lls[s1, a, t], transition_ll + s_lls[s2, t + 1]
                        )

    # Max-reduce to get state/action ML trellises for conveience
    s_lls = np.max(sa_lls, axis=1)

    # Identify our starting time
    if np.isneginf(np.max(s_lls[start])):
        # There is no feasible path from s1 to sg less or equal to than max_path_length
        return None
    start_time = np.argmax(s_lls[start, :])

    # Walk forward from start state, start time to re-construct path
    state = start
    time = start_time
    ml_path = []
    while state != goal:
        action = np.argmax(sa_lls[state, :, time])
        ml_path.append((state, action))
        successor_states = [s for (a, s) in xtr.children[state] if a == action]

        # Choose successor state with highest log likelihood at time + 1
        ml = -np.inf
        next_state = None
        for s2 in successor_states:
            s2_ll = s_lls[s2, time + 1]
            if s2_ll >= ml:
                next_state = s2
                ml = s2_ll

        state = next_state
        time = time + 1

    # Add final (goal) state
    ml_path.append((state, None))

    return ml_path


def main():
    """Main function"""
    pass


if __name__ == "__main__":
    main()
