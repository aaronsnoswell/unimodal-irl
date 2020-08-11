"""Metrics for evaluating IRL algorithm performance"""

import numpy as np

from unimodal_irl.rl_soln import (
    value_iteration,
    q_from_v,
    EpsilonGreedyPolicy,
    policy_evaluation,
)


def ile_evd(
    env_GT,
    env_IRL,
    *,
    p=1,
    optimal_policy_value=None,
    optimal_policy=None,
    verbose=False,
    vi_kwargs={},
    pi_kwargs={}
):
    """Find Inverse Learning Error and Expected Valud Difference metrics
    
    Inverse Learning Error is defined in "Inverse reinforcement learning in partially
    observable environments." by Choi and Kim, 2011.
    
    Expected Value Difference is defined in "Nonlinear inverse reinforcement learning
    with gaussian processes." by Levine, et al. 2011. EVD is essentially a weighted
    version of ILE, that only considers states with non-zero starting probability.
    
    Args:
        env_GT (unimodal_irl.envs.explicit.IExplicitEnv) Environment with GT reward
        env_IRL (unimodal_irl.envs.explicit.IExplicitEnv) Environment with learned
            reward
        
        p (int): p-Norm to use for ILE, Choi and Kim and other papers recommend p=1
        optimal_policy_value (numpy array): Optional shortcut - provide a pre-computed
            value array for the optimal policy to save computing it in this function
        optimal_policy (object): Optional shortcut - provide a pre-trained optimal
            policy so we don't have to compute it
        verbose (bool): Extra logging info
        vi_kwargs (dict): Extra keyword args for value_iteration
        pi_kwargs (dict): Extra keyword args for policy_iteration
    
    Returns:
        (float): Inverse Learning Error metric
        (float): Expected Value Difference metric
    """
    # Find the ground truth value of the optimal policy
    if verbose:
        print("Solving for GT value of optimal policy")
    if optimal_policy_value is None:
        if optimal_policy is None:
            v_GT = value_iteration(env_GT, **vi_kwargs)
            q_GT = q_from_v(v_GT, env_GT)
            pi_GT = EpsilonGreedyPolicy(q_GT)
        else:
            pi_GT = optimal_policy
        vpi_GT = policy_evaluation(env_GT, pi_GT, **pi_kwargs)
    else:
        vpi_GT = optimal_policy_value

    if verbose:
        print("Optimal policy GT value = \n{}".format(vpi_GT))

    # Find the value of the optimal policy for the learned reward, when evaluated on
    # the true reward
    if verbose:
        print("Solving for GT value of learned policy")
    v_IRL = value_iteration(env_IRL, **vi_kwargs)
    q_IRL = q_from_v(v_IRL, env_IRL)
    pi_IRL = EpsilonGreedyPolicy(q_IRL)
    vpi_IRL = policy_evaluation(env_GT, pi_IRL, **pi_kwargs)
    if verbose:
        print("Learned policy GT value = \n{}".format(vpi_IRL))

    ile = np.linalg.norm(vpi_GT - vpi_IRL, ord=p)
    evd = env_GT.p0s @ (vpi_GT - vpi_IRL)
    return ile, evd
