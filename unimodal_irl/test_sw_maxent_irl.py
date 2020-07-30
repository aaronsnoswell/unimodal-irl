"""Tests methods in sw_maxent_irl.py"""

import pytest
import numpy as np

# Test fixtures
from unimodal_irl.envs.explicit_linear import LinearMDPEnv
from unimodal_irl.envs.utils import pad_terminal_mdp

# Methods to test
from unimodal_irl.sw_maxent_irl import (
    backward_pass,
    backward_pass_log,
    forward_pass,
    forward_pass_log,
    partition,
    partition_log,
    marginals,
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


class TestBackward:
    """Test the backward message passing methods"""

    L = 6

    def test_backward_log_rs(self, env_rs):
        """Test log-space matches non-log-space w/ state reward function"""

        print("Testing backward message passing log calculation: state rewards")
        alpha = backward_pass(
            env_rs.p0s,
            TestBackward.L,
            env_rs.t_mat,
            env_rs.parents,
            gamma=env_rs.gamma,
            rs=env_rs.state_rewards,
        )
        alpha_log = backward_pass_log(
            env_rs.p0s,
            TestBackward.L,
            env_rs.t_mat,
            env_rs.parents,
            gamma=env_rs.gamma,
            rs=env_rs.state_rewards,
        )
        np.testing.assert_array_almost_equal(alpha, np.exp(alpha_log))

    def test_backward_log_rsa(self, env_rsa):
        """Test log-space matches non-log-space w/ state-action reward function"""

        print("Testing backward message passing log calculation: state-action rewards")

        alpha = backward_pass(
            env_rsa.p0s,
            TestBackward.L,
            env_rsa.t_mat,
            env_rsa.parents,
            gamma=env_rsa.gamma,
            rsa=env_rsa.state_action_rewards,
        )
        alpha_log = backward_pass_log(
            env_rsa.p0s,
            TestBackward.L,
            env_rsa.t_mat,
            env_rsa.parents,
            gamma=env_rsa.gamma,
            rsa=env_rsa.state_action_rewards,
        )
        np.testing.assert_array_almost_equal(alpha, np.exp(alpha_log))

    def test_backward_log_rsas(self, env_rsas):
        """Test log-space matches non-log-space w/ state-action-state reward function"""

        print(
            "Testing backward message passing log calculation: state-action-state rewards"
        )

        alpha = backward_pass(
            env_rsas.p0s,
            TestBackward.L,
            env_rsas.t_mat,
            env_rsas.parents,
            gamma=env_rsas.gamma,
            rsas=env_rsas.state_action_state_rewards,
        )
        alpha_log = backward_pass_log(
            env_rsas.p0s,
            TestBackward.L,
            env_rsas.t_mat,
            env_rsas.parents,
            gamma=env_rsas.gamma,
            rsas=env_rsas.state_action_state_rewards,
        )
        np.testing.assert_array_almost_equal(alpha, np.exp(alpha_log))


class TestForward:
    """Test the forward message passing methods"""

    L = 6

    def test_forward_log_rs(self, env_rs):
        """Test log-space matches non-log-space w/ state reward function"""

        print("Testing forward message passing log calculation: state rewards")

        beta = forward_pass(
            TestForward.L,
            env_rs.t_mat,
            env_rs.children,
            gamma=env_rs.gamma,
            rs=env_rs.state_rewards,
        )

        beta_log = forward_pass_log(
            TestForward.L,
            env_rs.t_mat,
            env_rs.children,
            gamma=env_rs.gamma,
            rs=env_rs.state_rewards,
        )
        np.testing.assert_array_almost_equal(beta, np.exp(beta_log))

    def test_forward_log_rsa(self, env_rsa):
        """Test log-space matches non-log-space w/ state-action reward function"""

        print("Testing forward message passing log calculation: state-action rewards")

        beta = forward_pass(
            TestForward.L,
            env_rsa.t_mat,
            env_rsa.children,
            gamma=env_rsa.gamma,
            rsa=env_rsa.state_action_rewards,
        )

        beta_log = forward_pass_log(
            TestForward.L,
            env_rsa.t_mat,
            env_rsa.children,
            gamma=env_rsa.gamma,
            rsa=env_rsa.state_action_rewards,
        )
        np.testing.assert_array_almost_equal(beta, np.exp(beta_log))

    def test_forward_log_rsas(self, env_rsas):
        """Test log-space matches non-log-space w/ state-action-state reward function"""

        print(
            "Testing forward message passing log calculation: state-action-state rewards"
        )

        beta = forward_pass(
            TestForward.L,
            env_rsas.t_mat,
            env_rsas.children,
            gamma=env_rsas.gamma,
            rsas=env_rsas.state_action_state_rewards,
        )

        beta_log = forward_pass_log(
            TestForward.L,
            env_rsas.t_mat,
            env_rsas.children,
            gamma=env_rsas.gamma,
            rsas=env_rsas.state_action_state_rewards,
        )
        np.testing.assert_array_almost_equal(beta, np.exp(beta_log))


class TestPartition:
    """Test Partition computation methods"""

    def test_partition_log(self, env_all):
        """Test log-space matches non-log-space w/ full reward function"""

        print("Testing partition log calculation")

        L = 6

        alpha = backward_pass(
            env_all.p0s,
            L,
            env_all.t_mat,
            env_all.parents,
            gamma=env_all.gamma,
            rs=env_all.state_rewards,
            rsa=env_all.state_action_rewards,
            rsas=env_all.state_action_state_rewards,
        )

        Z_theta = partition(L, alpha)
        with np.errstate(divide="ignore"):
            Z_theta_log = partition_log(L, np.log(alpha))
        np.testing.assert_almost_equal(Z_theta, np.exp(Z_theta_log))


class TestMarginals:
    """Test Marginal computations"""

    L = 6

    def test_marginals_log(self, env_all):
        """Test log-space matches non-log-space w/ full reward function"""

        print("Testing marginal log calculation")

        alpha = backward_pass(
            env_all.p0s,
            TestMarginals.L,
            env_all.t_mat,
            env_all.parents,
            gamma=env_all.gamma,
            rs=env_all.state_rewards,
            rsa=env_all.state_action_rewards,
            rsas=env_all.state_action_state_rewards,
        )
        beta = forward_pass(
            TestMarginals.L,
            env_all.t_mat,
            env_all.children,
            gamma=env_all.gamma,
            rs=env_all.state_rewards,
            rsa=env_all.state_action_rewards,
            rsas=env_all.state_action_state_rewards,
        )

        Z_theta = partition(TestMarginals.L, alpha)

        pts, ptsa, ptsas = marginals(
            TestMarginals.L,
            env_all.t_mat,
            alpha,
            beta,
            Z_theta,
            env_all.gamma,
            rsa=env_all.state_action_rewards,
            rsas=env_all.state_action_state_rewards,
        )

        # Suppress possible log(0) warnings
        with np.errstate(divide="ignore"):
            alpha_log = np.log(alpha)
            beta_log = np.log(beta)
            Z_theta_log = np.log(Z_theta)

        pts_log, ptsa_log, ptsas_log = marginals_log(
            TestMarginals.L,
            env_all.t_mat,
            alpha_log,
            beta_log,
            Z_theta_log,
            env_all.gamma,
            rsa=env_all.state_action_rewards,
            rsas=env_all.state_action_state_rewards,
        )

        np.testing.assert_array_almost_equal(pts, np.exp(pts_log))
        np.testing.assert_array_almost_equal(ptsa, np.exp(ptsa_log))
        np.testing.assert_array_almost_equal(ptsas, np.exp(ptsas_log))

    def _get_GT_values(self, linear_mdp_env):
        """Compute ground truth values for the Linear MDP example
        
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

        # Partition value
        z_GT = ltau1 + ltau2 + ltau3 + ltau4

        # Path probabilities
        ptau1_GT = ltau1 / z_GT
        ptau2_GT = ltau2 / z_GT
        ptau3_GT = ltau3 / z_GT
        ptau4_GT = ltau4 / z_GT

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

    def test_marginals_GT(self, env_all):
        """Test that marginals are correctly computed"""

        print("Testing marginal with respect to ground truth")

        alpha = backward_pass(
            env_all.p0s,
            TestMarginals.L,
            env_all.t_mat,
            env_all.parents,
            gamma=env_all.gamma,
            rs=env_all.state_rewards,
            rsa=env_all.state_action_rewards,
            rsas=env_all.state_action_state_rewards,
        )

        beta = forward_pass(
            TestMarginals.L,
            env_all.t_mat,
            env_all.children,
            gamma=env_all.gamma,
            rs=env_all.state_rewards,
            rsa=env_all.state_action_rewards,
            rsas=env_all.state_action_state_rewards,
        )

        Z_theta = partition(TestMarginals.L, alpha)

        pts, ptsa, ptsas = marginals(
            TestMarginals.L,
            env_all.t_mat,
            alpha,
            beta,
            Z_theta,
            gamma=env_all.gamma,
            rsa=env_all.state_action_rewards,
            rsas=env_all.state_action_state_rewards,
        )

        # Drop dummy components
        pts = pts[0:-1, 0:4]
        ptsa = ptsa[0:-1, 0:-1, 0:3]
        ptsas = ptsas[0:-1, 0:-1, 0:-1, 0:3]

        # Compute ground truth values for the Linear MDP
        pts_GT, ptsa_GT, ptsas_GT = self._get_GT_values(env_all)

        np.testing.assert_array_almost_equal(pts, pts_GT)
        np.testing.assert_array_almost_equal(ptsa, ptsa_GT)
        np.testing.assert_array_almost_equal(ptsas, ptsas_GT)
