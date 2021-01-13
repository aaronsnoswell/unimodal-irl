"""Metrics for evaluating IRL algorithm performance"""

import numpy as np

from mdp_extras import v_vi, v2q, OptimalPolicy, pi_eval


def ile_evd(
    xtr,
    phi,
    reward_gt,
    reward_test,
    *,
    p=1,
    vi_kwargs={},
    policy_kwargs={},
    pe_kwargs={},
    ret_gt_value=False,
    gt_policy_value=None
):
    """Find Inverse Learning Error and Expected Value Difference metrics
    
    Inverse Learning Error is defined in "Inverse reinforcement learning in partially
    observable environments." by Choi and Kim, 2011.
    
    Expected Value Difference is defined in "Nonlinear inverse reinforcement learning
    with gaussian processes." by Levine, et al. 2011. EVD is essentially a weighted
    version of ILE, that only considers states with non-zero starting probability.
    
    Args:
        xtr (mdp_extras.DiscreteExplicitExtras) MDP extras object
        phi (mdp_extras.FeatureFunction) Feature function for MDP
        reward_gt (mdp_extras.RewardFunction): Ground Truth reward function for MDP
        reward_test (mdp_extras.RewardFunction): Learned reward function for MDP
        
        p (int): p-Norm to use for ILE, Choi and Kim and other papers recommend p=1
        vi_kwargs (dict): Extra keyword args for mdp_extras.v_vi Value Iteration method
        policy_kwargs (dict): Extra keyword args for mdp_extras.OptimalPolicy
        pe_kwargs (dict): Extra keyword args for mdp_extras.pi_eval Policy Evaluation method
        ret_gt_value (bool): If true, also return the GT policy state value function,
            can be used for speeding up future calls
        gt_policy_value (numpy array): Optional ground truth policy state value function
            - used for speeding up this function with multiple calls
    
    Returns:
        (float): Inverse Learning Error metric
        (float): Expected Value Difference metric
    """
    
    if gt_policy_value is None:
        # Get GT policy state value function
        gt_policy_value = v_vi(xtr, phi, reward_gt, **vi_kwargs)

    # Get test policy state value function under GT reward
    v_star_test = v_vi(xtr, phi, reward_test, **vi_kwargs)
    q_star = v2q(v_star_test, xtr, phi, reward_test)
    pi_star_test = OptimalPolicy(q_star, stochastic=False, **policy_kwargs)
    test_policy_value = pi_eval(xtr, phi, reward_gt, pi_star_test, **pe_kwargs)

    value_delta = gt_policy_value - test_policy_value
    ile = np.linalg.norm(value_delta, ord=p)
    evd = xtr.p0s @ value_delta
    
    if not ret_gt_value:
        return ile, evd
    else:
        return ile, evd, gt_policy_value
