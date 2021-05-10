"""Implements Maximum Causal Entropy by Ziebart et al."""

import numpy as np


def softmax(x, t=1.0):
    """Numerically stable soft maximum of a vector.

    This is computed by considering the vector elements pairwise, and exploiting the fact that

    log(exp(a) + exp(b) + exp(c)) = log(exp(log(exp(a) + exp(b)) + exp(c))

    Args:
        x (numpy array): Vector of values to take soft maximum of

        t (float): Temperature parameter > 0. As t -> 0, this resembles a regular max.

    Returns:
        (numpy array): Soft-maximum of the elements of x, or t * log(\sum_i exp(x_i/t))
    """

    x = np.array(x)
    if x.shape == ():
        # Handle edge-case of a single scalar
        return x

    # Compute sm of each arg one by one
    val = x[0]
    for x_i in x[1:]:
        max_val = max(val, x_i)
        min_val = min(val, x_i)
        delta = min_val - max_val
        # Numerically stable softmax of two arguments
        val = max_val + t * np.log(1 + np.exp(delta / t))
    return val


def mellowmax(x, t=1.0):
    """Numerically stable mellow maximum of a vector

    Unlike softmax, mellowmax is a contraction mapping, which means it has a unique fixed point.

    Args:
        x (numpy array): Vector of values to take mellow maximum of

        t (float): Temperature parameter. As t -> 0, this resembles a regular max.

    Returns:
        (numpy array): Mellow-maximum of the elements of x, or t * log(1/n \sum_i^n exp(x_i/t))
    """
    x = np.array(x)
    if x.shape == ():
        # Handle edge-case of a single scalar
        return x
    return softmax(x, t) - t * np.log(len(x))


if __name__ == "__main__":
    print(mellowmax([1.0, 1.0, 1.0, 35.0], t=1.0))
    print("Here")
