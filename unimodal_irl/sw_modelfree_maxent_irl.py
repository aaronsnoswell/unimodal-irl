import numpy as np

from mdp_extras import log_sum_exp

#
# def sw_modelfree_maxent_irl(
#     x,
#     gamma,
#     phi,
#     phi_bar,
#     pi_ref,
#     get_ref_demos,
#     max_path_length,
#     min_path_length=1,
#     nll_only=False,
#     true_logZ=None,
# ):
#     """Model-free Maximum Entropy IRL`
#
#     Args:
#         x (numpy array):
#         gamma (float): MDP discount factor
#         phi (FeatureFunction): Feature function for the MDP
#         phi_bar (numpy array): Feature expectation under expert policy
#         pi_ref (Policy): Reference policy used for importance sampling
#         get_ref_demos (callable): Function that outputs an list of of (s, a)
#             rollouts sampled from the environment under pi_ref
#         max_path_length (int): Maximum path length allowed in partition calculation
#
#         min_path_length (int): Minimum path length allowed in partition calculation
#         nll_only (bool): If true, compute NLL only, not gradient (faster)
#     """
#
#     pi_ref_demos = get_ref_demos()
#     N = len(pi_ref_demos)
#
#     num_path_lengths = max_path_length - min_path_length + 1
#
#     # Compute the 'x' values that need log-sum-exp-ing
#     path_log_likelihoods = np.array(
#         [
#             x @ phi.onpath(d, gamma)
#             + np.log(num_path_lengths)
#             - pi_ref.path_log_action_probability(d)
#             for d in pi_ref_demos
#         ]
#     )
#     log_Z_theta = log_sum_exp(path_log_likelihoods) - np.log(N)
#
#     # Compute NLL
#     nll = log_Z_theta - x @ phi_bar
#     if nll_only:
#         return nll
#
#     # Compute gradient
#     feature_vectors = np.array([phi.onpath(d, gamma) for d in pi_ref_demos])
#     if len(feature_vectors) == 1:
#         efv = feature_vectors[0]
#     else:
#         # log_weights = []
#         # phi_taus = []
#         # for d_idx, d in enumerate(pi_ref_demos):
#         #     phi_tau_prime = phi.onpath(d, gamma)
#         #     phi_taus.append(phi_tau_prime)
#         #     log_weights.append(
#         #         x @ phi_tau_prime
#         #         + np.log(num_path_lengths)
#         #         - pi_ref.path_log_action_probability(d)
#         #     )
#         #
#         # # Subtract max - biased estimate
#         # log_weights = np.array(log_weights)
#         # log_weights -= np.max(log_weights)
#         # weights = np.exp(log_weights)
#         # weights /= np.sum(weights)
#         #
#         # phi_taus = np.array(phi_taus)
#         # efv = np.average(phi_taus, axis=0, weights=weights)
#
#         efv = np.zeros(len(phi))
#         for d_idx, d in enumerate(pi_ref_demos):
#             phi_tau_prime = phi.onpath(d, gamma)
#             efv += (
#                 num_path_lengths
#                 * np.exp(
#                     x @ phi_tau_prime
#                     - true_logZ
#                     - pi_ref.path_log_action_probability(d)
#                 )
#                 * phi_tau_prime
#             )
#         efv /= N
#
#     nll_grad = efv - phi_bar
#
#     return nll, nll_grad


def sw_maxent_irl_modelfree(
    x, xtr, phi, phi_bar, max_path_length, pi_ref, pi_ref_demos, nll_only=False
):
    """Compute MaxEnt IRL negative log likelihood with importance sampling

    Args:

    Returns:
        (float): Un-based negative log likelihood estimate for reward parameter x
    """

    M = len(pi_ref_demos)
    z = max_path_length

    # Compute NLL
    fis = np.array(
        [
            np.log(z)
            + x @ phi.onpath(pi_ref_demo, xtr.gamma)
            - pi_ref.path_log_action_probability(pi_ref_demo)
            for pi_ref_demo in pi_ref_demos
        ]
    )
    log_Z_theta = log_sum_exp(fis) - np.log(M)
    nll = log_Z_theta - x @ phi_bar

    if nll_only:
        return nll

    # Also compute NLL gradient estimate
    F = log_sum_exp(fis)
    pis = np.exp(fis - F)
    grad_log_Z_theta = np.sum(
        [pi * phi.onpath(pi_ref_demo) for pi, pi_ref_demo in zip(pis, pi_ref_demos)],
        axis=0,
    )
    nll_grad = grad_log_Z_theta - phi_bar

    return nll, nll_grad


def main():
    """Main function"""

    import gym
    import warnings

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from tqdm import tqdm
    from scipy.optimize import minimize

    from mdp_extras import (
        vi,
        OptimalPolicy,
        padding_trick,
        UniformRandomPolicy,
        PaddedMDPWarning,
        Linear,
    )
    from mdp_extras.envs import nchain_extras, frozen_lake_extras

    from unimodal_irl import sw_maxent_irl, sw_modelfree_maxent_irl, mean_ci, ile_evd

    n = 5
    env = gym.make("NChain-v0", n=n)
    xtr, phi, reward_gt = nchain_extras(env, gamma=0.9)

    _, q_star = vi(xtr, phi, reward_gt)
    pi_star = OptimalPolicy(q_star)
    max_path_length = 10

    num_paths = 40
    demo_star = pi_star.get_rollouts(env, num_paths, max_path_length=max_path_length)
    phi_bar = phi.demo_average(demo_star, xtr.gamma)

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=PaddedMDPWarning)
        # Compute ground truth values
        gt_nll, gt_grad = sw_maxent_irl(
            reward_gt.theta, xtr, phi, phi_bar, max_path_length
        )

    print(f"GT: {gt_nll:.3f} {gt_grad}")

    pi_ref = UniformRandomPolicy(len(xtr.actions))

    # num_reference_paths = 20
    for num_reference_paths in 2 ** np.arange(13):

        nll_errs = []
        grad_errs = []
        for rep in range(100):

            pi_ref_demos = []
            for _ in range(num_reference_paths):
                path_len = np.random.randint(1, max_path_length + 1)
                pi_ref_demos.extend(
                    pi_ref.get_rollouts(env, 1, max_path_length=path_len)
                )

            nll, grad = sw_maxent_irl_modelfree(
                reward_gt.theta,
                xtr,
                phi,
                phi_bar,
                max_path_length,
                pi_ref,
                pi_ref_demos,
                nll_only=False,
            )
            # print(f"IS: {nll:.3f} {grad}")
            nll_err = np.sqrt((nll - gt_nll) ** 2)
            grad_err = np.linalg.norm(gt_grad - grad)
            nll_errs.append(nll_err)
            grad_errs.append(grad_err)

        print(
            f"IS ({num_reference_paths}): {np.mean(nll_err):.3f} {np.mean(grad_err):.3f}"
        )

    print("Howedy")


if __name__ == "__main__":
    main()
