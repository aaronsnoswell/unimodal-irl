
# Uni-Modal IRL

Algorithms for uni-modal Inverse Reinforcement Learning.

This package provides several optimized uni-modal IRL algorithm reference
implementations.

*Uni-modal* here refers to the assumption of a single expert demonstrating a single
task - as compared to [*multi-modal*](https://github.com/aaronsnoswell/multimodal-irl)
IRL (sometimes known as *multi-task* IRL).

## Installation

This package is not distributed on PyPI - you'll have to install from source.

```bash
git clone https://github.com/aaronsnoswell/unimodal-irl.git
cd unimodal-irl
pip install -e .
```

## Usage

Each `*_irl.py` file provides one IRL algorithm as python method.
These methods return the optimization objective and/or gradient and are designed for
use with optimizers such as `scipy.optimize.minimize()`.

These methods accept parameters that are based on the [`mdp-extras`](https://github.com/aaronsnoswell/mdp-extras)
helper library.
Please see the documentation for that library to understand how to construct the
appropriate parameters for your problem.

For example, to run Maximum Likelihood IRL on the `FrozenLake` MDP;

```python
import gym
import numpy as np

from scipy.optimize import minimize

from mdp_extras import q_vi, OptimalPolicy, padding_trick
from mdp_extras.envs import frozen_lake_extras

from unimodal_irl import sw_maxent_irl

# Get FrozenLake MDP environment object
env = gym.make("FrozenLake-v0")

# Convert to explicit dynamics, features, and rewards using mdp_extras library
xtr, phi, reward = frozen_lake_extras(env, gamma=0.99)

# Find optimal Q(s, a) function
q_star = q_vi(xtr, phi, reward)

# Find optimal stationary stochastic policy pi(a | s)
pi_star = OptimalPolicy(q_star, stochastic=True)

# Sample 10 rollouts from the optimal policy
rollouts = pi_star.get_rollouts(env, 10)

# Apply padding trick to make all demonstrations the same length
xtr, phi, reward, rollouts = padding_trick(xtr, phi, reward, rollouts)

# Initial reward parameter estimate
theta0 = np.zeros(len(phi)) + np.mean(reward.range)

# Min and max values for reward parameter vector
theta_bounds = tuple(reward.range for _ in range(len(phi)))

# Solve for maximum likelihood reward function parameters
res = minimize(
    sw_maxent_irl,
    theta0,
    args=(xtr, phi, rollouts, True),
    method="L-BFGS-B",
    jac=True,
    bounds=theta_bounds,
    options=dict(disp=True),
)
print(res)

# Get optimal reward parameters (drop the dummy state that was added by the padding trick)
theta_star = res.x[:-1]
print(theta_star.reshape(4, 4))

print("Done")
```

### Exact Maximum Entropy Inverse Reinforcement Learning (ExactMaxEntIRL)

The file [`sw_maxent_irl.py`](unimodal_irl/sw_maxent_irl.py) implements the exact
Maximum Entropy IRL algorithm 'ExactMaxEntIRL' by Snoswell et al., 2020.

 * *Revisiting Maximum Entropy Inverse Reinforcement Learning: New Perspectives and Algorithms*
   by Snoswell, A. J., Singh, S. P. N. and Ye, N., In IEEE SSCI, 2020

The top level function is `sw_maxent_irl()`

### Approximate Maximum Entropy Inverse Reinforcement Learning

The file [`zb_maxent_irl`](unimodal_irl/zb_maxent_irl.py) implements the approximate
Maximum Entropy IRL algorithm 'Algorithm 1' by Ziebart et al., 2008.

 * [*Maximum Entropy Inverse Reinforcement Learning*](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)
   by B. Ziebart, A. Mass, J. A. Bagnell and A. K. Dey, In AAAI, 2008

The top-level function is `zb_maxent_irl()` with the parameter `version='08'`.

This paper, and the algorithm, were updated in 2010 to handle terminal states in a
slightly different manner.

 * [*Maximum Entropy Inverse Reinforcement Learning* (manuscript updated)](http://www-cgi.cs.cmu.edu/afs/cs.cmu.edu/Web/People/bziebart/publications/maxentirl-bziebart.pdf) by B. Ziebart, A. Mass, J. A.
   Bagnell and A. K. Dey, AAAI, 2010

The top-level function is `zb_maxent_irl()` with the parameter `version='10'`
(the default value).

### Maximum Likelihood Inverse Reinforcement Learning

The file [`bv_maxlikleihood_irl.py`](unimodal_irl/bv_maxlikleihood_irl.py) implements
the Maximum Likelihood IRL algorithm 'Algorithm 1' by Babeş-Vroman et al., 2011.

 * [*Apprenticeship learning about multiple intentions*](https://icml.cc/2011/papers/478_icmlpaper.pdf)
   by Babes, M., Marivate, V. N., Subramanian, K., & Littman, M. L., In ICML, 2011

The top level function is `bv_maxlikleihood_irl()`

