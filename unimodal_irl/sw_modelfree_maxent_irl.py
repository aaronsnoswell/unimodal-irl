import numpy as np

from mdp_extras import log_sum_exp


def sw_modelfree_maxent_irl(
    x, gamma, phi, phi_bar, pi_ref, get_ref_demos, nll_only=False
):
    """Model-free Maximum Entropy IRL
    
    Args:
        x (numpy array):
        gamma (float): MDP discount factor
        phi (FeatureFunction): Feature function for the MDP
        phi_bar (numpy array): Feature expectation under expert policy
        pi_ref (Policy): Reference policy used for importance sampling
        get_ref_demos (callable): Function that outputs an list of of (s, a)
            rollouts sampled from the environment under pi_ref

        nll_only (bool): If true, compute NLL only, not gradient (faster)
    """

    pi_ref_rollouts = get_ref_demos()
    N = len(pi_ref_rollouts)

    # Compute the 'x' values that need log-sum-exp-ing
    path_log_likelihoods = np.array(
        [
            x @ phi.onpath(r, gamma) - pi_ref.path_log_action_probability(r)
            for r in pi_ref_rollouts
        ]
    )
    log_Z_theta = log_sum_exp(path_log_likelihoods) - np.log(N)

    # Compute NLL
    nll = log_Z_theta - x @ phi_bar
    if nll_only:
        return nll

    # Compute gradient
    feature_vectors = np.array([phi.onpath(r, gamma) for r in pi_ref_rollouts])
    if len(feature_vectors) == 1:
        efv = feature_vectors[0]
    else:
        weights = np.exp(path_log_likelihoods - log_Z_theta - np.log(N))
        efv = np.zeros(len(phi_bar))
        for w, r in zip(weights, pi_ref_rollouts):
            efv += w * phi.onpath(r, gamma)

    nll_grad = efv - phi_bar

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
        q_vi,
        OptimalPolicy,
        padding_trick,
        UniformRandomPolicy,
        PaddedMDPWarning,
        Linear,
    )
    from mdp_extras.envs import nchain_extras

    from unimodal_irl import sw_maxent_irl, sw_modelfree_maxent_irl, mean_ci, ile_evd

    env = gym.make("NChain-v0")
    xtr, phi, reward_gt = nchain_extras(env, gamma=0.9)

    q_star = q_vi(xtr, phi, reward_gt)
    pi_star = OptimalPolicy(q_star)
    max_path_length = 50

    num_paths = 400
    demo_star = pi_star.get_rollouts(env, num_paths, max_path_length=max_path_length)
    phi_bar = phi.demo_average(demo_star, xtr.gamma)
    xtr_p, demo_star_p = padding_trick(xtr, demo_star)

    pi_ref = UniformRandomPolicy(len(xtr.actions))

    num_reference_paths = 10
    get_ref_demos = lambda: pi_ref.get_rollouts(
        env, num_reference_paths, max_path_length=max_path_length
    )
    theta = np.random.randn(len(phi))

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=PaddedMDPWarning)
        gt_nll, gt_grad = sw_maxent_irl(
            theta, xtr_p, phi, phi_bar, max_path_length=max_path_length
        )
        print("GT Z grad is ")
        print(gt_grad + phi_bar)

        nll, grad = sw_modelfree_maxent_irl(
            theta, xtr.gamma, phi, phi_bar, pi_ref, get_ref_demos
        )

    pass


if __name__ == "__main__":
    main()
