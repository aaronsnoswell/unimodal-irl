
# Uni-Modal IRL

Algorithms for uni-modal Inverse Reinforcement Learning.

This package provides several optimized uni-modal IRL algorithm reference
implementations.

## Installation

This package is not distributed on PyPI - you'll have to install from source.

```bash
git clone https://github.com/aaronsnoswell/unimodal-irl.git
cd unimodal-irl
pip install -e .
```

To run the included [unit-tests](unimodal_irl/test_sw_maxent_irl.py),

```bash
cd unimodal-irl
pytest .
```

## Usage

### Exact Maximum Entropy Inverse Reinforcement Learning (ExactMaxEntIRL)

The file [`sw_maxent_irl.py`](unimodal_irl/sw_maxent_irl.py) implements the exact
Maximum Entropy IRL algorithm 'ExactMaxEntIRL' by Snoswell et al., 2020 (paper under
review).

The top level function is `sw_maxent_irl()`

### Approximate Maximum Entropy Inverse Reinforcement Learning

The file [`zb_maxent_irl`](unimodal_irl/zb_maxent_irl.py) implements the approximate
Maximum Entropy IRL algorithm 'Algorithm 1' by Ziebart et al., 2008.

 * [*Maximum Entropy Inverse Reinforcement Learning*](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf) by B. Ziebart, A. Mass, J. A.
   Bagnell and A. K. Dey, AAAI, 2008

The top-level function is `zb_maxent_irl()` with the parameter `version='08'`.

This paper, and the algorithm, were updated in 2010 to handle terminal states in a
slightly different manner.

 * [*Maximum Entropy Inverse Reinforcement Learning* (manuscript updated)](http://www-cgi.cs.cmu.edu/afs/cs.cmu.edu/Web/People/bziebart/publications/maxentirl-bziebart.pdf) by B. Ziebart, A. Mass, J. A.
   Bagnell and A. K. Dey, AAAI, 2010

The top-level function is `zb_maxent_irl()` with the parameter `version='10'`
(the default value).
