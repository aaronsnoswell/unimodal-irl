"""Tests methods in sw_maxent_irl.py"""

import pytest
import numpy as np

# Test fixtures
from unimodal_irl.envs.explicit_linear import LinearMDPEnv
from unimodal_irl.envs.utils import pad_terminal_mdp

# Methods to test
from unimodal_irl.sw_maxent_irl import (
    backward_pass_log,
    forward_pass_log,
    partition_log,
    marginals_log,
)


@pytest.fixture
def env_rs():
    """Prepare environment with state rewards only"""
    np.random.seed(0)
    env = LinearMDPEnv()

    # Overload rewards for testing purposes
    env._state_rewards = np.random.randn(4)

    # Pad the MDP
    env = pad_terminal_mdp(env)

    return env


@pytest.fixture
def env_rsa():
    """Prepare environment with state-action rewards only"""
    np.random.seed(0)
    env = LinearMDPEnv()

    # Overload rewards for testing purposes
    env._state_action_rewards = np.random.randn(4, 1)

    # Pad the MDP
    env = pad_terminal_mdp(env)

    return env


@pytest.fixture
def env_rsas():
    """Prepare environment with state-action-state rewards only"""
    np.random.seed(0)
    env = LinearMDPEnv()

    # Overload rewards for testing purposes
    env._state_action_state_rewards = np.random.randn(4, 1, 4)

    # Pad the MDP
    env = pad_terminal_mdp(env)

    return env


@pytest.fixture
def env_all():
    """Prepare environment with state, state-action, and state-action-state rewards"""
    np.random.seed(0)
    env = LinearMDPEnv()

    # Overload rewards for testing purposes
    env._state_rewards = np.random.randn(4)
    env._state_action_rewards = np.random.randn(4, 1)
    env._state_action_state_rewards = np.random.randn(4, 1, 4)

    # Pad the MDP
    env = pad_terminal_mdp(env)

    return env


class TestMarginals:
    """Test Marginal computations"""

    L = 6

    def _get_GT_likelihoods(self, linear_mdp_env):
        """Compute ground truth path likelihoods for the Linear MDP example
        
        These values are computed using manually derived and verified equations
        specific to this example MDP
        """
        # Path likelihoods
        ltau1 = np.exp(linear_mdp_env.state_rewards[0])
        ltau2 = np.exp(
            linear_mdp_env.state_rewards[0]
            + linear_mdp_env.state_action_rewards[0, 0]
            + linear_mdp_env.state_action_state_rewards[0, 0, 1]
            + linear_mdp_env.gamma * (linear_mdp_env.state_rewards[1])
        )
        ltau3 = np.exp(
            linear_mdp_env.state_rewards[0]
            + linear_mdp_env.state_action_rewards[0, 0]
            + linear_mdp_env.state_action_state_rewards[0, 0, 1]
            + linear_mdp_env.gamma
            * (
                linear_mdp_env.state_rewards[1]
                + linear_mdp_env.state_action_rewards[1, 0]
                + linear_mdp_env.state_action_state_rewards[1, 0, 2]
                + linear_mdp_env.gamma * (linear_mdp_env.state_rewards[2])
            )
        )
        ltau4 = np.exp(
            linear_mdp_env.state_rewards[0]
            + linear_mdp_env.state_action_rewards[0, 0]
            + linear_mdp_env.state_action_state_rewards[0, 0, 1]
            + linear_mdp_env.gamma
            * (
                linear_mdp_env.state_rewards[1]
                + linear_mdp_env.state_action_rewards[1, 0]
                + linear_mdp_env.state_action_state_rewards[1, 0, 2]
                + linear_mdp_env.gamma
                * (
                    linear_mdp_env.state_rewards[2]
                    + linear_mdp_env.state_action_rewards[2, 0]
                    + linear_mdp_env.state_action_state_rewards[2, 0, 3]
                    + linear_mdp_env.gamma * (linear_mdp_env.state_rewards[3])
                )
            )
        )
        return ltau1, ltau2, ltau3, ltau4

    def _get_GT_partition(self, linear_mdp_env):
        """Compute ground truth partition value for the Linear MDP example
        
        These values are computed using manually derived and verified equations
        specific to this example MDP
        """
        ltau1_GT, ltau2_GT, ltau3_GT, ltau4_GT = self._get_GT_likelihoods(
            linear_mdp_env
        )

        # Partition value
        z_GT = ltau1_GT + ltau2_GT + ltau3_GT + ltau4_GT

        return z_GT

    def _get_GT_probs(self, linear_mdp_env):
        """Compute ground truth path probabilities for the Linear MDP example
        
        These values are computed using manually derived and verified equations
        specific to this example MDP
        """
        ltau1, ltau2, ltau3, ltau4 = self._get_GT_likelihoods(linear_mdp_env)
        z_GT = self._get_GT_partition(linear_mdp_env)

        # Path probabilities
        ptau1_GT = ltau1 / z_GT
        ptau2_GT = ltau2 / z_GT
        ptau3_GT = ltau3 / z_GT
        ptau4_GT = ltau4 / z_GT

        return ptau1_GT, ptau2_GT, ptau3_GT, ptau4_GT

    def _get_GT_marginals(self, linear_mdp_env):
        """Compute ground truth marginals for the Linear MDP example
        
        These values are computed using manually derived and verified equations
        specific to this example MDP
        """

        ptau1_GT, ptau2_GT, ptau3_GT, ptau4_GT = self._get_GT_probs(linear_mdp_env)

        # State marginals
        pts_GT = np.zeros((4, 4))
        pts_GT[0, 0] = ptau1_GT + ptau2_GT + ptau3_GT + ptau4_GT
        pts_GT[1, 1] = ptau2_GT + ptau3_GT + ptau4_GT
        pts_GT[2, 2] = ptau3_GT + ptau4_GT
        pts_GT[3, 3] = ptau4_GT

        # State-action marginals
        ptsa_GT = np.zeros((4, 1, 3))
        ptsa_GT[0, 0, 0] = ptau2_GT + ptau3_GT + ptau4_GT
        ptsa_GT[1, 0, 1] = ptau3_GT + ptau4_GT
        ptsa_GT[2, 0, 2] = ptau4_GT

        ptsas_GT = np.zeros((4, 1, 4, 3))
        ptsas_GT[0, 0, 1, 0] = ptau2_GT + ptau3_GT + ptau4_GT
        ptsas_GT[1, 0, 2, 1] = ptau3_GT + ptau4_GT
        ptsas_GT[2, 0, 3, 2] = ptau4_GT

        return pts_GT, ptsa_GT, ptsas_GT

    def _get_GT_alpha(self, linear_mdp_env):
        """Compute ground truth backward message values for the Linear MDP example
        
        These values are computed using manually derived and verified equations
        specific to this example MDP
        """

        ltau1_GT, ltau2_GT, ltau3_GT, ltau4_GT = self._get_GT_likelihoods(
            linear_mdp_env
        )

        alpha_GT = np.zeros((4, TestMarginals.L))
        alpha_GT[0, 0] = ltau1_GT
        alpha_GT[1, 1] = ltau2_GT
        alpha_GT[2, 2] = ltau3_GT
        alpha_GT[3, 3] = ltau4_GT

        return alpha_GT

    def _get_GT_beta(self, linear_mdp_env):
        """Compute ground truth forward message values for the Linear MDP example
        
        These values are computed using manually derived and verified equations
        specific to this example MDP
        """

        ltau1_GT, ltau2_GT, ltau3_GT, ltau4_GT = self._get_GT_likelihoods(
            linear_mdp_env
        )

        beta_GT = np.zeros((4, TestMarginals.L))

        raise NotImplementedError

        return beta_GT

    def test_alpha_GT(self, env_all):
        """Test that the partition is correctly computed"""

        print("Testing backward message with respect to ground truth")

        alpha_log = backward_pass_log(
            env_all.p0s,
            TestMarginals.L,
            env_all.t_mat,
            env_all.parents,
            gamma=env_all.gamma,
            rs=env_all.state_rewards,
            rsa=env_all.state_action_rewards,
            rsas=env_all.state_action_state_rewards,
        )

        # Drop dummy state
        alpha_log = alpha_log[:-1, :]

        # Compute ground truth backward message for Linear MDP
        alpha_GT = self._get_GT_alpha(env_all)

        np.testing.assert_array_almost_equal(np.exp(alpha_log), alpha_GT)

    def test_beta_GT(self, env_all):
        """Test that the partition is correctly computed"""

        print("Testing forward message with respect to ground truth")

        beta_log = forward_pass_log(
            TestMarginals.L,
            env_all.t_mat,
            env_all.children,
            gamma=env_all.gamma,
            rs=env_all.state_rewards,
            rsa=env_all.state_action_rewards,
            rsas=env_all.state_action_state_rewards,
        )

        # Drop dummy state
        beta_log = beta_log[:-1, :]

        # Compute ground truth backward message for Linear MDP
        beta_GT = self._get_GT_beta(env_all)

        with np.printoptions(linewidth=999):
            print(np.exp(beta_log))
            print(beta_GT)

        np.testing.assert_array_almost_equal(np.exp(beta_log), beta_GT)

    def test_partition_GT(self, env_all):
        """Test that the partition is correctly computed"""

        print("Testing marginal with respect to ground truth")

        alpha_log = backward_pass_log(
            env_all.p0s,
            TestMarginals.L,
            env_all.t_mat,
            env_all.parents,
            gamma=env_all.gamma,
            rs=env_all.state_rewards,
            rsa=env_all.state_action_rewards,
            rsas=env_all.state_action_state_rewards,
        )
        Z_theta_log = partition_log(TestMarginals.L, alpha_log)

        # Compute ground truth partition for the Linear MDP
        z_GT = self._get_GT_partition(env_all)

        np.testing.assert_almost_equal(np.exp(Z_theta_log), z_GT)

    def test_marginals_GT(self, env_all):
        """Test that marginals are correctly computed"""

        print("Testing marginal with respect to ground truth")

        alpha_log = backward_pass_log(
            env_all.p0s,
            TestMarginals.L,
            env_all.t_mat,
            env_all.parents,
            gamma=env_all.gamma,
            rs=env_all.state_rewards,
            rsa=env_all.state_action_rewards,
            rsas=env_all.state_action_state_rewards,
        )

        beta_log = forward_pass_log(
            TestMarginals.L,
            env_all.t_mat,
            env_all.children,
            gamma=env_all.gamma,
            rs=env_all.state_rewards,
            rsa=env_all.state_action_rewards,
            rsas=env_all.state_action_state_rewards,
        )

        Z_theta_log = partition_log(TestMarginals.L, alpha_log)

        pts_log, ptsa_log, ptsas_log = marginals_log(
            TestMarginals.L,
            env_all.t_mat,
            alpha_log,
            beta_log,
            Z_theta_log,
            gamma=env_all.gamma,
            rsa=env_all.state_action_rewards,
            rsas=env_all.state_action_state_rewards,
        )

        # Drop dummy components
        pts_log = pts_log[0:-1, 0:4]
        ptsa_log = ptsa_log[0:-1, 0:-1, 0:3]
        ptsas_log = ptsas_log[0:-1, 0:-1, 0:-1, 0:3]

        # Compute ground truth marginals for the Linear MDP
        pts_GT, ptsa_GT, ptsas_GT = self._get_GT_marginals(env_all)

        np.testing.assert_array_almost_equal(np.exp(pts_log), pts_GT)
        np.testing.assert_array_almost_equal(np.exp(ptsa_log), ptsa_GT)
        np.testing.assert_array_almost_equal(np.exp(ptsas_log), ptsas_GT)
