"""Various utility methods related to IRL algorithms"""


def incremental_mean(x, prev_mean, k):
    """Compute a mean incrementally over a dataset of num items

    This is useful when we might run out of memory storing all the individual components of the mean

    Args:
        x (any): The current (k-th) datapoint
        prev_mean (any): The previous datapoint. If set to None, then we assert k == 1
        k (int): The index of the current datapoint, starting from 1
    """
    if prev_mean is None:
        assert k == 1
        return x
    else:
        diff = x - prev_mean
        return prev_mean + diff / k
