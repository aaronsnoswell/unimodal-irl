"""Metrics for evaluating IRL algorithm performance"""

import numpy as np

from unimodal_irl.rl_soln import (
    value_iteration,
    q_from_v,
    EpsilonGreedyPolicy,
    policy_evaluation,
)


def ile(env_GT, env_IRL, *, p=1, optimal_policy=None, verbose=False):
    """Find Inverse Learning Error metric
    
    Inverse Learning Error is defined in "Inverse reinforcement learning in partially
    observable environments." by Choi and Kim, 2011.
    
    Args:
        env_GT (unimodal_irl.envs.explicit.IExplicitEnv) Environment with GT reward
        env_IRL (unimodal_irl.envs.explicit.IExplicitEnv) Environment with learned
            reward
        
        p (int): p-Norm to use, Choi and Kim and other papers recommend p=1
        optimal_policy (object): Optional shortcut - provide a pre-trained optimal
            policy so we don't have to compute it
        verbose (bool): Extra logging info
    
    Returns:
        (float): Inverse Learning Error
    """
    # Find the ground truth value of the optimal policy
    if verbose:
        print("Solving for GT value of optimal policy")
    if optimal_policy is None:
        v_GT = value_iteration(env_GT, verbose=verbose)
        q_GT = q_from_v(v_GT, env_GT)
        pi_GT = EpsilonGreedyPolicy(q_GT)
    else:
        pi_GT = optimal_policy
    vpi_GT = policy_evaluation(env_GT, pi_GT)
    if verbose:
        print("Optimal policy GT value = \n{}".format(vpi_GT))

    # Find the value of the optimal policy for the learned reward, when evaluated on
    # the true reward
    if verbose:
        print("Solving for GT value of learned policy")
    v_IRL = value_iteration(env_IRL, verbose=verbose)
    q_IRL = q_from_v(v_IRL, env_IRL)
    pi_IRL = EpsilonGreedyPolicy(q_IRL)
    vpi_IRL = policy_evaluation(env_GT, pi_IRL)
    if verbose:
        print("Learned policy GT value = \n{}".format(vpi_IRL))

    return np.linalg.norm(vpi_GT - vpi_IRL, ord=p)


def evd(env_GT, env_IRL, optimal_policy=None, verbose=False):
    """Find Expected Value Difference metric
    
    Expected Value Difference is defined in "Nonlinear inverse reinforcement learning
    with gaussian processes." by Levine, et al. 2011.
    
    Args:
        env_GT (unimodal_irl.envs.explicit.IExplicitEnv) Environment with GT reward
        env_IRL (unimodal_irl.envs.explicit.IExplicitEnv) Environment with learned
            reward
        
        optimal_policy (object): Optional shortcut - provide a pre-trained optimal
            policy so we don't have to compute it
        verbose (bool): Extra logging information
    
    Returns:
        (float): Inverse Learning Error
    """
    # Find the ground truth value of the optimal policy
    if verbose:
        print("Solving for GT value of optimal policy")
    if optimal_policy is None:
        v_GT = value_iteration(env_GT, verbose=verbose)
        q_GT = q_from_v(v_GT, env_GT)
        pi_GT = EpsilonGreedyPolicy(q_GT)
    else:
        pi_GT = optimal_policy
    vpi_GT = policy_evaluation(env_GT, pi_GT)
    if verbose:
        print("Optimal policy GT value = \n{}".format(vpi_GT))

    # Find the value of the optimal policy for the learned reward, when evaluated on
    # the true reward
    if verbose:
        print("Solving for GT value of learned policy")
    v_IRL = value_iteration(env_IRL, verbose=verbose)
    q_IRL = q_from_v(v_IRL, env_IRL)
    pi_IRL = EpsilonGreedyPolicy(q_IRL)
    vpi_IRL = policy_evaluation(env_GT, pi_IRL)
    if verbose:
        print("Learned policy GT value = \n{}".format(vpi_IRL))

    # Expected Value Difference is the expectation under starting distribution of the
    # difference in GT and learned policy value under the true reward
    return env_GT.p0s @ (vpi_GT - vpi_IRL)
