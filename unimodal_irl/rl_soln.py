"""Find optimal Value and Policy functions for explicit Gym environments
"""

import pickle
import warnings
import numpy as np
import itertools as it


def value_iteration(env, eps=1e-6, verbose=False, max_iter=None):
    """Value iteration to find the optimal value function
    
    Args:
        env (.envs.explicit_env.IExplicitEnv) Explicit Gym environment
        
        eps (float): Value convergence tolerance
        verbose (bool): Extra logging
        max_iter (int): If provided, iteration will terminate regardless of convergence
            after this many iterations.
    
    Returns:
        (numpy array): |S| vector of state values
    """

    if env.gamma == 1.0:
        warnings.warn(
            "Environment discount factor is 1.0 - value iteration will only converge if all paths terminate in a finite number of steps"
        )

    value_fn = np.zeros(env.t_mat.shape[0])

    # Prepare linear reward arrays
    _state_rewards = env.state_rewards
    if _state_rewards is None:
        _state_rewards = np.zeros(env.t_mat.shape[0])
    _state_action_rewards = env.state_action_rewards
    if _state_action_rewards is None:
        _state_action_rewards = np.zeros(env.t_mat.shape[0 : 1 + 1])
    _state_action_state_rewards = env.state_action_state_rewards
    if _state_action_state_rewards is None:
        _state_action_state_rewards = np.zeros(env.t_mat.shape)

    for _iter in it.count():
        delta = 0
        for s1 in env.states:
            v = value_fn[s1]
            value_fn[s1] = np.max(
                [
                    np.sum(
                        [
                            env.t_mat[s1, a, s2]
                            * (
                                _state_action_rewards[s1, a]
                                + _state_action_state_rewards[s1, a, s2]
                                + _state_rewards[s2]
                                + env.gamma * value_fn[s2]
                            )
                            for s2 in env.states
                        ]
                    )
                    for a in env.actions
                ]
            )
            delta = max(delta, np.abs(v - value_fn[s1]))

        if max_iter is not None and _iter >= max_iter:
            if verbose:
                print("Terminating before convergence at {} iterations".format(_iter))
                break

        # Check value function convergence
        if delta < eps:
            break
        else:
            if verbose:
                print("Value Iteration #{}, delta={}".format(_iter, delta))

    return value_fn


def q_from_v(v_star, env):
    """Find Q* given V*
    
    Args:
        v_star (numpy array): |S| vector of optimal state values
        env (.envs.explicit_env.IExplicitEnv) Explicit Gym environment
    
    Returns:
        (numpy array): |S|x|A| array of optimal state-action values
    """

    # Prepare linear reward arrays
    _state_rewards = env.state_rewards
    if _state_rewards is None:
        _state_rewards = np.zeros(env.t_mat.shape[0])
    _state_action_rewards = env.state_action_rewards
    if _state_action_rewards is None:
        _state_action_rewards = np.zeros(env.t_mat.shape[0 : 1 + 1])
    _state_action_state_rewards = env.state_action_state_rewards
    if _state_action_state_rewards is None:
        _state_action_state_rewards = np.zeros(env.t_mat.shape)

    q_star = np.zeros(env.t_mat.shape[0 : 1 + 1])

    for s1, a, s2 in it.product(env.states, env.actions, env.states):
        q_star[s1, a] += env.t_mat[s1, a, s2] * (
            _state_action_rewards[s1, a]
            + _state_action_state_rewards[s1, a, s2]
            + _state_rewards[s2]
            + env.gamma * v_star[s2]
        )

    return q_star


class EpsilonGreedyPolicy:
    """An Epsilon Greedy Policy wrt. a provided Q function
    
    Provides a .predict(s) method to match the stable-baselines policy API
    """

    def __init__(self, q, epsilon=0.0):
        """C-tor
        
        Args:
            q (numpy array): |S|x|A| Q-matrix
            epsilon (float): Probability of taking a random action
        """
        self.q = q
        self.epsilon = epsilon
        self.optimal_action_map = {s: np.argmax(q[s]) for s in range(q.shape[0])}

    def save(self, path):
        """Save policy to file"""
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        """Load policy from file"""
        with open(path, "rb") as file:
            _self = pickle.load(file)
            return _self

    def prob_for_state(self, s):
        """Get the epsilon greedy action probability vector for a given state"""
        p = np.ones(self.q.shape[1]) * (self.epsilon) / self.q.shape[1]
        p[self.optimal_action_map[s]] += 1 - self.epsilon
        return p

    def predict(self, s):
        """Predict next action and distribution over states"""
        action = None

        if self.epsilon == 0.0:
            action = self.optimal_action_map[s]
        else:
            action = np.random.choice(
                np.arange(self.q.shape[1]), p=self.prob_for_state[s]
            )

        return action, None


def policy_evaluation(env, policy, tolerance=1e-6):
    """Determine the value function of a given deterministic policy
    
    Args:
        env (.envs.explicit_env.IExplicitEnv) Explicit Gym environment
        policy (object): Policy object providing a deterministic .predict(s) method to
            match the stable-baselines policy API
        
        tolerance (float): State value convergence threshold
    
    Returns:
        (numpy array): |S| state value vector
    """
    v_pi = np.zeros_like(env.states, dtype=float)

    # Prepare linear reward arrays
    _state_rewards = env.state_rewards
    if _state_rewards is None:
        _state_rewards = np.zeros(env.t_mat.shape[0])
    _state_action_rewards = env.state_action_rewards
    if _state_action_rewards is None:
        _state_action_rewards = np.zeros(env.t_mat.shape[0 : 1 + 1])
    _state_action_state_rewards = env.state_action_state_rewards
    if _state_action_state_rewards is None:
        _state_action_state_rewards = np.zeros(env.t_mat.shape)

    for _iteration in it.count():
        delta = 0

        for s1 in env.states:
            v = v_pi[s1]
            v_pi[s1] = np.sum(
                [
                    float(policy.predict(s1)[0] == a)
                    * np.sum(
                        [
                            env.t_mat[s1, a, s2]
                            * (
                                _state_action_rewards[s1, a]
                                + _state_action_state_rewards[s1, a, s2]
                                + _state_rewards[s2]
                                + env.gamma * v_pi[s2]
                            )
                            for s2 in env.states
                        ]
                    )
                    for a in env.actions
                ]
            )
            delta = max(delta, np.abs(v - v_pi[s1]))

        if delta < tolerance:
            break

    return v_pi
