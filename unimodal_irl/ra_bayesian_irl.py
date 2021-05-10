import numpy as np
import scipy as sp


def policywalk():
    pass


def main():

    import gym
    import random
    import itertools as it

    from mdp_extras import (
        vi,
        OptimalPolicy,
        Indicator,
        Linear,
        BoltzmannExplorationPolicy,
    )
    from mdp_extras.envs import nchain_extras

    n = 2
    env = gym.make("NChain-v0", n=n)
    xtr, phi, reward_gt = nchain_extras(env, gamma=0.9)

    rollouts = OptimalPolicy(vi(xtr, phi, reward_gt)[1]).get_rollouts(
        env, 10, max_path_length=10
    )

    mean = np.zeros_like(reward_gt.theta)
    r_prior = sp.stats.multivariate_normal(mean, 5.0)

    r = r_prior.rvs(1)
    _, q = vi(xtr, phi, Linear(r))
    pi = OptimalPolicy(q, stochastic=False)
    print(r.reshape(n, -1))

    num_acceptances = 0
    for i in it.count():
        r_tilde = sp.stats.multivariate_normal(r).rvs(1)
        _, q_tilde = vi(xtr, phi, Linear(r_tilde))

        pi_actions = np.argmax([pi.prob_for_state(s) for s in xtr.states], axis=1)
        q_actions = np.argmax(q_tilde, axis=1)

        if not np.all(pi_actions == q_actions):
            # We've reached a new policy equivalence class

            pib_tilde = BoltzmannExplorationPolicy(q_tilde)
            pib = BoltzmannExplorationPolicy(q)
            logprob_tilde = 0.0
            logprob = 0.0
            for p in rollouts:
                logprob_tilde += pib_tilde.path_log_action_probability(p)
                logprob += pib.path_log_action_probability(p)
            logprob_tilde += r_prior.logpdf(r_tilde)
            logprob += r_prior.logpdf(r)

            acceptance_logprob = logprob_tilde - logprob
            acceptance_prob = np.exp(acceptance_logprob)
            acceptance_prob = min(1, acceptance_prob)

            if np.random.rand() <= acceptance_prob:
                # Accept proposal
                r = r_tilde
                q = q_tilde
                pi = OptimalPolicy(q_tilde)
                num_acceptances += 1
                print(f"Accept ({num_acceptances / (i+1) * 100}%)")
                print(r.reshape(n, -1))
            else:
                # Reject
                print(f"Reject ({num_acceptances / (i+1) * 100}%)")
                continue


if __name__ == "__main__":
    main()
