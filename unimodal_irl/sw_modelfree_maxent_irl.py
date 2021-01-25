import numpy as np

from mdp_extras import UniformRandomPolicy


def sw_modelfree_maxent_irl(
    x, gamma, phi, phi_bar, pi_ref, pi_ref_rollouts, nll_only=False
):
    """Model-free Maximum Entropy IRL
    
    Args:
        x (numpy array):
        gamma (float): MDP discount factor
        phi (FeatureFunction): Feature function for the MDP
        phi_bar (numpy array): Feature expectation under expert policy
        pi_ref (Policy): Reference policy used for importance sampling
        pi_ref_rollouts (list): List of (s, a) rollouts sampled from the reference
            policy
        
        nll_only (bool): If true, compute NLL only, not gradient (faster)
    """

    weighted_path_likelihoods = []
    path_fvs = []
    for r in pi_ref_rollouts:
        rollout_feature_vec = phi.expectation([r], gamma=xtr.gamma)
        path_fvs.append(rollout_feature_vec)

        importance_weight = 1.0 / np.exp(pi_ref.path_log_likelihood(r))
        path_likelihood = np.exp(x @ rollout_feature_vec)
        weighted_path_likelihoods.append(importance_weight * path_likelihood)

    weighted_path_likelihoods = np.array(weighted_path_likelihoods)
    path_fvs = np.array(path_fvs)
    Z_theta = np.sum(weighted_path_likelihoods)

    # Compute NLL
    nll = np.log(Z_theta) - x @ phi_bar
    if nll_only:
        return nll

    # Compute gradient
    weighted_path_probs = weighted_path_likelihoods / Z_theta
    efv = np.average(path_fvs, axis=0, weights=weighted_path_probs)
    nll_grad = efv - phi_bar

    return nll, nll_grad


def main():
    """Main function"""
    pass


if __name__ == "__main__":
    main()
