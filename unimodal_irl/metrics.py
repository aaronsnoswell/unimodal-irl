"""Metrics for evaluating IRL algorithm performance"""

import warnings

import numpy as np
from scipy.stats import norm

from mdp_extras import vi, v2q, OptimalPolicy, pi_eval


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
    gt_policy_value=None,
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
        vi_kwargs (dict): Extra keyword args for mdp_extras.vi Value Iteration method
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
        gt_policy_value, _ = vi(xtr, phi, reward_gt, **vi_kwargs)

    # Get test policy state value function under GT reward
    v_star_test, q_star_test = vi(xtr, phi, reward_test, **vi_kwargs)
    pi_star_test = OptimalPolicy(q_star_test, stochastic=False, **policy_kwargs)
    test_policy_value = pi_eval(xtr, phi, reward_gt, pi_star_test, **pe_kwargs)

    value_delta = gt_policy_value - test_policy_value
    ile = np.linalg.norm(value_delta, ord=p)
    evd = xtr.p0s @ value_delta

    if evd < 0:
        warnings.warn(
            f"EVD is < 0 (by {0 - evd}) - possible loss of accuracy due to numerical rounding"
        )
        evd = 0.0
    if ile < 0:
        warnings.warn(
            f"ILE is < 0 (by {0 - ile}) - possible loss of accuracy due to numerical rounding"
        )
        ile = 0.0

    if not ret_gt_value:
        return ile, evd
    else:
        return ile, evd, gt_policy_value


def inner_angle(v1, v2):
    """Find inner angle between two n-vectors in radians

    Args:
        v1 (numpy array): First vector
        v2 (numpy array): Second vector

    Returns:
        (float): Inner angle in radians
    """

    assert len(v1.shape) == 1
    assert v1.shape == v2.shape

    return np.arccos(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def mean_ci(values, confidence_level=0.95):
    """Compute mean and symmetric confidence interval for a list of values

    Args:
        values (list): List of float
        confidence_level (float): Confidence level

    Returns:
        (float): Lower confidence interval
        (float): Mean value
        (float): Upper confidence interval
    """
    mean = np.mean(values)
    std = np.std(values)
    num_repeats = len(values)

    # Compute Z-factor
    critical_value = 1.0 - confidence_level
    z_factor = norm().ppf(1 - critical_value / 2)

    # Compute CI
    ci = z_factor * std / np.sqrt(num_repeats)

    return mean - ci, mean, mean + ci


def mean_ci_agg(values, *args):
    """Aggregator for use with Pandas groupby() function

    Args:
        values (pandas DataFrame): DataFrame with floating point values to apply aggregator to

        *args: Optional args to pass to mean_ci function

    Returns:
        (pandas DataFrame): DataFrame containing strings of the form "Mean ± CI"
    """
    if np.any(values.isna().to_numpy()):
        return "N/A"
    low, med, high = mean_ci(values, *args)
    interval = high - low
    return "{:03.2f} $\pm$ {:05.2f}".format(med, interval)


def median_ci(values, confidence_level=0.95):
    """Compute median and approximate confidence interval for a list of values

    The method of computing the CI is taken from
    https://www.ucl.ac.uk/child-health/short-courses-events/about-statistical-courses/research-methods-and-statistics/chapter-8-content-8

    Args:
        values (list): List of float
        confidence_level (float): Confidence level

    Returns:
        (float): Lower approximate confidence interval
        (float): Median value
        (float): Upper approximate confidence interval
    """
    median = np.median(values)
    num_repeats = len(values)

    # Compute Z-factor
    critical_value = 1.0 - confidence_level
    z_factor = norm().ppf(1 - critical_value / 2)

    # Compute CI rankings
    low_ci_rank = int(round(num_repeats / 2 - z_factor * np.sqrt(num_repeats) / 2))
    high_ci_rank = int(round(1 + num_repeats / 2 + z_factor * np.sqrt(num_repeats) / 2))
    values_sorted = sorted(values)

    return values_sorted[low_ci_rank], median, values_sorted[high_ci_rank]


def median_ci_agg(values, *args):
    """Aggregator for use with Pandas groupby() function

    Args:
        values (pandas DataFrame): DataFrame with floating point values to apply aggregator to

        *args: Optional args to pass to median_ci function

    Returns:
        (pandas DataFrame): DataFrame containing strings of the form "Median [Low - High]"
    """
    if np.any(values.isna().to_numpy()):
        return "N/A"
    low, med, high = median_ci(values, *args)
    return "{:.2f} [{:.2f} - {:.2f}]".format(low, med, high)
