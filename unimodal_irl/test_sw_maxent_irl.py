"""Tests methods in sw_maxent_irl.py"""

import pytest
import numpy as np

from unimodal_irl.envs.linear_mdp import LinearMDPEnv
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
from unimodal_irl.utils import pad_mdp, compute_parents_children


@pytest.fixture
def env_rs():
    """Prepare environment with state rewards only"""
    np.random.seed(0)
    env = LinearMDPEnv()
    reward_scale = 1
    env.state_rewards = np.random.randn(4) * reward_scale
    env = pad_mdp(env)
    return env


@pytest.fixture
def env_rsa():
    """Prepare environment with state and state-action rewards"""
    np.random.seed(0)
    env = LinearMDPEnv()
    reward_scale = 1
    env.state_rewards = np.random.randn(4) * reward_scale
    env.state_action_rewards = np.random.randn(4, 1) * reward_scale
    env = pad_mdp(env)
    return env


@pytest.fixture
def env_rsas():
    """Prepare environment with state, state-action, and state-action-state rewards"""
    np.random.seed(0)
    env = LinearMDPEnv()
    reward_scale = 1
    env.state_rewards = np.random.randn(4) * reward_scale
    env.state_action_rewards = np.random.randn(4, 1) * reward_scale
    env.state_action_state_rewards = np.random.randn(4, 1, 4) * reward_scale
    env = pad_mdp(env)
    return env


class TestBackward:

    L = 6
    gamma = 0.9

    def test_backward_log_rs(self, env_rs):
        print("Testing backward message passing log calculation: state rewards")
        parents, children = compute_parents_children(
            env_rs.t, env_rs.terminal_state_mask
        )
        alpha = backward_pass(
            env_rs.p0s,
            TestBackward.L,
            env_rs.t,
            parents,
            gamma=TestBackward.gamma,
            rs=env_rs.state_rewards,
        )
        alpha_log = backward_pass_log(
            env_rs.p0s,
            TestBackward.L,
            env_rs.t,
            parents,
            gamma=TestBackward.gamma,
            rs=env_rs.state_rewards,
        )
        np.testing.assert_array_almost_equal(alpha, np.exp(alpha_log))

    def test_backward_log_rsa(self, env_rsa):
        print("Testing backward message passing log calculation: state-action rewards")
        parents, children = compute_parents_children(
            env_rsa.t, env_rsa.terminal_state_mask
        )
        alpha = backward_pass(
            env_rsa.p0s,
            TestBackward.L,
            env_rsa.t,
            parents,
            gamma=TestBackward.gamma,
            rs=env_rsa.state_rewards,
            rsa=env_rsa.state_action_rewards,
        )
        alpha_log = backward_pass_log(
            env_rsa.p0s,
            TestBackward.L,
            env_rsa.t,
            parents,
            gamma=TestBackward.gamma,
            rs=env_rsa.state_rewards,
            rsa=env_rsa.state_action_rewards,
        )
        np.testing.assert_array_almost_equal(alpha, np.exp(alpha_log))

    def test_backward_log_rsas(self, env_rsas):
        print(
            "Testing backward message passing log calculation: state-action-state rewards"
        )
        parents, children = compute_parents_children(
            env_rsas.t, env_rsas.terminal_state_mask
        )
        alpha = backward_pass(
            env_rsas.p0s,
            TestBackward.L,
            env_rsas.t,
            parents,
            gamma=TestBackward.gamma,
            rs=env_rsas.state_rewards,
            rsa=env_rsas.state_action_rewards,
            rsas=env_rsas.state_action_state_rewards,
        )
        alpha_log = backward_pass_log(
            env_rsas.p0s,
            TestBackward.L,
            env_rsas.t,
            parents,
            gamma=TestBackward.gamma,
            rs=env_rsas.state_rewards,
            rsa=env_rsas.state_action_rewards,
            rsas=env_rsas.state_action_state_rewards,
        )
        np.testing.assert_array_almost_equal(alpha, np.exp(alpha_log))


class TestForward:

    L = 6
    gamma = 0.9

    def test_forward_log_rs(self, env_rs):
        print("Testing forward message passing log calculation: state rewards")
        parents, children = compute_parents_children(
            env_rs.t, env_rs.terminal_state_mask
        )
        beta = forward_pass(
            TestForward.L,
            env_rs.t,
            children,
            gamma=TestForward.gamma,
            rs=env_rs.state_rewards,
        )

        beta_log = forward_pass_log(
            TestForward.L,
            env_rs.t,
            children,
            gamma=TestForward.gamma,
            rs=env_rs.state_rewards,
        )
        np.testing.assert_array_almost_equal(beta, np.exp(beta_log))

    def test_forward_log_rsa(self, env_rsa):
        print("Testing forward message passing log calculation: state-action rewards")
        parents, children = compute_parents_children(
            env_rsa.t, env_rsa.terminal_state_mask
        )
        beta = forward_pass(
            TestForward.L,
            env_rsa.t,
            children,
            gamma=TestForward.gamma,
            rs=env_rsa.state_rewards,
            rsa=env_rsa.state_action_rewards,
        )

        beta_log = forward_pass_log(
            TestForward.L,
            env_rsa.t,
            children,
            gamma=TestForward.gamma,
            rs=env_rsa.state_rewards,
            rsa=env_rsa.state_action_rewards,
        )
        np.testing.assert_array_almost_equal(beta, np.exp(beta_log))

    def test_forward_log_rsas(self, env_rsas):
        print(
            "Testing forward message passing log calculation: state-action-state rewards"
        )
        parents, children = compute_parents_children(
            env_rsas.t, env_rsas.terminal_state_mask
        )
        beta = forward_pass(
            TestForward.L,
            env_rsas.t,
            children,
            gamma=TestForward.gamma,
            rs=env_rsas.state_rewards,
            rsa=env_rsas.state_action_rewards,
            rsas=env_rsas.state_action_state_rewards,
        )

        beta_log = forward_pass_log(
            TestForward.L,
            env_rsas.t,
            children,
            gamma=TestForward.gamma,
            rs=env_rsas.state_rewards,
            rsa=env_rsas.state_action_rewards,
            rsas=env_rsas.state_action_state_rewards,
        )
        np.testing.assert_array_almost_equal(beta, np.exp(beta_log))


def test_partition_log(env_rsas):
    print("Testing partition log calculation")
    L = 6
    gamma = 0.9
    parents, children = compute_parents_children(
        env_rsas.t, env_rsas.terminal_state_mask
    )
    alpha = backward_pass(
        env_rsas.p0s,
        L,
        env_rsas.t,
        parents,
        gamma=gamma,
        rs=env_rsas.state_rewards,
        rsa=env_rsas.state_action_rewards,
        rsas=env_rsas.state_action_state_rewards,
    )

    Z_theta = partition(L, alpha)
    with np.errstate(divide="ignore"):
        Z_theta_log = partition_log(L, np.log(alpha))
    np.testing.assert_almost_equal(Z_theta, np.exp(Z_theta_log))


class TestMarginals:

    L = 6
    gamma = 0.9

    def test_marginals_log(self, env_rsas):
        print("Testing marginal log calculation")
        parents, children = compute_parents_children(
            env_rsas.t, env_rsas.terminal_state_mask
        )
        alpha = backward_pass(
            env_rsas.p0s,
            TestMarginals.L,
            env_rsas.t,
            parents,
            gamma=TestMarginals.gamma,
            rs=env_rsas.state_rewards,
        )
        beta = forward_pass(
            TestMarginals.L,
            env_rsas.t,
            children,
            gamma=TestMarginals.gamma,
            rs=env_rsas.state_rewards,
            rsa=env_rsas.state_action_rewards,
            rsas=env_rsas.state_action_state_rewards,
        )

        Z_theta = partition(TestMarginals.L, alpha)

        pts, ptsa, ptsas = marginals(
            TestMarginals.L,
            env_rsas.t,
            alpha,
            beta,
            Z_theta,
            TestMarginals.gamma,
            rsa=env_rsas.state_action_rewards,
            rsas=env_rsas.state_action_state_rewards,
        )

        with np.errstate(divide="ignore"):
            pts_log, ptsa_log, ptsas_log = marginals_log(
                TestMarginals.L,
                env_rsas.t,
                np.log(alpha),
                np.log(beta),
                np.log(Z_theta),
                TestMarginals.gamma,
                rsa=env_rsas.state_action_rewards,
                rsas=env_rsas.state_action_state_rewards,
            )

        np.testing.assert_array_almost_equal(pts, np.exp(pts_log))
        np.testing.assert_array_almost_equal(ptsa, np.exp(ptsa_log))
        np.testing.assert_array_almost_equal(ptsas, np.exp(ptsas_log))

    def test_marginals_GT(self, env_rsas):
        print("Testing marginal with respect to ground truth")

        # Compute ground truth values for the Linear MDP
        # Path likelihoods
        ltau1 = np.exp(env_rsas.state_rewards[0])
        ltau2 = np.exp(
            env_rsas.state_rewards[0]
            + env_rsas.state_action_rewards[0, 0]
            + env_rsas.state_action_state_rewards[0, 0, 1]
            + TestMarginals.gamma * (env_rsas.state_rewards[1])
        )
        ltau3 = np.exp(
            env_rsas.state_rewards[0]
            + env_rsas.state_action_rewards[0, 0]
            + env_rsas.state_action_state_rewards[0, 0, 1]
            + TestMarginals.gamma
            * (
                env_rsas.state_rewards[1]
                + env_rsas.state_action_rewards[1, 0]
                + env_rsas.state_action_state_rewards[1, 0, 2]
                + TestMarginals.gamma * (env_rsas.state_rewards[2])
            )
        )
        ltau4 = np.exp(
            env_rsas.state_rewards[0]
            + env_rsas.state_action_rewards[0, 0]
            + env_rsas.state_action_state_rewards[0, 0, 1]
            + TestMarginals.gamma
            * (
                env_rsas.state_rewards[1]
                + env_rsas.state_action_rewards[1, 0]
                + env_rsas.state_action_state_rewards[1, 0, 2]
                + TestMarginals.gamma
                * (
                    env_rsas.state_rewards[2]
                    + env_rsas.state_action_rewards[2, 0]
                    + env_rsas.state_action_state_rewards[2, 0, 3]
                    + TestMarginals.gamma * (env_rsas.state_rewards[3])
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

        # Compute children and parent dictionaries
        parents, children = compute_parents_children(
            env_rsas.t, env_rsas.terminal_state_mask
        )

        alpha = backward_pass(
            env_rsas.p0s,
            TestMarginals.L,
            env_rsas.t,
            parents,
            gamma=TestMarginals.gamma,
            rs=env_rsas.state_rewards,
            rsa=env_rsas.state_action_rewards,
            rsas=env_rsas.state_action_state_rewards,
        )

        beta = forward_pass(
            TestMarginals.L,
            env_rsas.t,
            children,
            gamma=TestMarginals.gamma,
            rs=env_rsas.state_rewards,
            rsa=env_rsas.state_action_rewards,
            rsas=env_rsas.state_action_state_rewards,
        )

        Z_theta = partition(TestMarginals.L, alpha)

        pts, ptsa, ptsas = marginals(
            TestMarginals.L,
            env_rsas.t,
            alpha,
            beta,
            Z_theta,
            TestMarginals.gamma,
            rsa=env_rsas.state_action_rewards,
            rsas=env_rsas.state_action_state_rewards,
        )

        # Drop dummy components
        pts = pts[0:-1, 0:4]
        ptsa = ptsa[0:-1, 0:-1, 0:3]
        ptsas = ptsas[0:-1, 0:-1, 0:-1, 0:3]

        np.testing.assert_array_almost_equal(pts, pts_GT)
        np.testing.assert_array_almost_equal(ptsa, ptsa_GT)
        np.testing.assert_array_almost_equal(ptsas, ptsas_GT)
