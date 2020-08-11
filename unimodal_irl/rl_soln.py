"""Find optimal Value and Policy functions for explicit Gym environments
"""

import pickle
import warnings
import numpy as np
import itertools as it

from numba import jit


def nonb_value_iteration(env, eps=1e-6, verbose=False, max_iter=None):
    """Value iteration to find the optimal value function
    
    Args:
        env (.envs.explicit.IExplicitEnv) Explicit Gym environment
        
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


def value_iteration(env, eps=1e-6, verbose=False, max_iter=None):
    """Value iteration to find the optimal value function
    
    Args:
        env (.envs.explicit.IExplicitEnv) Explicit Gym environment
        
        eps (float): Value convergence tolerance
        verbose (bool): Extra logging
        max_iter (int): If provided, iteration will terminate regardless of convergence
            after this many iterations.
    
    Returns:
        (numpy array): |S| vector of state values
    """

    # Numba has trouble typing these arguments to the forward/backward functions below.
    # We manually convert them here to avoid typing issues at JIT compile time
    rs = env.state_rewards
    if rs is None:
        rs = np.zeros(env.t_mat.shape[0], dtype=np.float)

    rsa = env.state_action_rewards
    if rsa is None:
        rsa = np.zeros(env.t_mat.shape[0:2], dtype=np.float)

    rsas = env.state_action_state_rewards
    if rsas is None:
        rsas = np.zeros(env.t_mat.shape[0:3], dtype=np.float)

    return _nb_value_iteration(
        env.t_mat, env.gamma, rs, rsa, rsas, eps=eps, verbose=verbose, max_iter=max_iter
    )


@jit(nopython=True)
def _nb_value_iteration(
    t_mat, gamma, rs, rsa, rsas, eps=1e-6, verbose=False, max_iter=None
):
    """Value iteration to find the optimal value function
    
    Args:
        eps (float): Value convergence tolerance
        verbose (bool): Extra logging
        max_iter (int): If provided, iteration will terminate regardless of convergence
            after this many iterations.
    
    Returns:
        (numpy array): |S| vector of state values
    """

    value_fn = np.zeros(t_mat.shape[0])

    _iter = 0
    while True:
        delta = 0
        for s1 in range(t_mat.shape[0]):
            v = value_fn[s1]
            action_values = np.zeros(t_mat.shape[1])
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    action_values[a] += t_mat[s1, a, s2] * (
                        rsa[s1, a] + rsas[s1, a, s2] + rs[s2] + gamma * value_fn[s2]
                    )
            value_fn[s1] = np.max(action_values)
            delta = max(delta, np.abs(v - value_fn[s1]))

        if max_iter is not None and _iter >= max_iter:
            if verbose:
                print("Terminating before convergence, # iterations = ", _iter)
                break

        # Check value function convergence
        if delta < eps:
            break
        else:
            if verbose:
                print("Value Iteration #", _iter, " delta=", delta)

        _iter += 1

    return value_fn


@jit(nopython=True)
def _nb_q_from_v(
    v_star,
    t_mat,
    gamma,
    state_rewards,
    state_action_rewards,
    state_action_state_rewards,
):
    """Find Q* given V* (numba optimized version)
    
    Args:
        v_star (numpy array): |S| vector of optimal state values
        t_mat (numpy array): |S|x|A|x|S| transition matrix
        gamma (float): Discount factor
        state_rewards (numpy array): |S| array of state rewards
        state_action_rewards (numpy array): |S|x|A| array of state-action rewards
        state_action_state_rewards (numpy array): |S|x|A|x|S| array of state-action-state rewards
    
    Returns:
        (numpy array): |S|x|A| array of optimal state-action values
    """

    q_star = np.zeros(t_mat.shape[0 : 1 + 1])

    for s1 in range(t_mat.shape[0]):
        for a in range(t_mat.shape[1]):
            for s2 in range(t_mat.shape[2]):
                q_star[s1, a] += t_mat[s1, a, s2] * (
                    state_action_rewards[s1, a]
                    + state_action_state_rewards[s1, a, s2]
                    + state_rewards[s2]
                    + gamma * v_star[s2]
                )

    return q_star


def q_from_v(v_star, env):
    """Find Q* given V*
    
    Args:
        v_star (numpy array): |S| vector of optimal state values
        env (.envs.explicit.IExplicitEnv) Explicit Gym environment
    
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

    return _nb_q_from_v(
        v_star,
        env.t_mat,
        env.gamma,
        _state_rewards,
        _state_action_rewards,
        _state_action_state_rewards,
    )


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


@jit(nopython=True)
def _nb_policy_evaluation(
    t_mat,
    gamma,
    state_rewards,
    state_action_rewards,
    state_action_state_rewards,
    policy_vector,
    eps=1e-6,
):
    """Determine the value function of a given deterministic policy
    
    Args:
        env (.envs.explicit.IExplicitEnv) Explicit Gym environment
        policy (object): Policy object providing a deterministic .predict(s) method to
            match the stable-baselines policy API
        
        eps (float): State value convergence threshold
    
    Returns:
        (numpy array): |S| state value vector
    """

    v_pi = np.zeros(t_mat.shape[0])

    _iteration = 0
    while True:
        delta = 0
        for s1 in range(t_mat.shape[0]):
            v = v_pi[s1]
            _tmp = 0
            for a in range(t_mat.shape[1]):
                if policy_vector[s1] != a:
                    continue
                for s2 in range(t_mat.shape[2]):
                    _tmp += t_mat[s1, a, s2] * (
                        state_action_rewards[s1, a]
                        + state_action_state_rewards[s1, a, s2]
                        + state_rewards[s2]
                        + gamma * v_pi[s2]
                    )
            v_pi[s1] = _tmp
            delta = max(delta, np.abs(v - v_pi[s1]))

        if delta < eps:
            break
        _iteration += 1

    return v_pi


def policy_evaluation(env, policy, eps=1e-6):
    """Determine the value function of a given deterministic policy
    
    Args:
        env (.envs.explicit.IExplicitEnv) Explicit Gym environment
        policy (object): Policy object providing a deterministic .predict(s) method to
            match the stable-baselines policy API
        
        eps (float): State value convergence threshold
    
    Returns:
        (numpy array): |S| state value vector
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

    return _nb_policy_evaluation(
        env.t_mat,
        env.gamma,
        _state_rewards,
        _state_action_rewards,
        _state_action_state_rewards,
        np.array([policy.predict(s)[0] for s in env.states]),
        eps=eps,
    )
