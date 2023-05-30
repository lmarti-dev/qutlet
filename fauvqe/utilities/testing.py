import collections
from typing import Iterable
import numpy as np


def non_zero_matrix(M, eq_tol: float = 1e-15) -> np.ndarray:
    """Returns an array with ones where the input array is non-null and zero elsewhere
    Args:
        M (np.ndarray): the input array
        eq_tol (float, optional): the tolerance below which a number is considered to be 0. Defaults to 1e-15.
    Returns:
        np.ndarray: the output array with only filled with ones and zeroes.
    """
    return (abs(np.array(M)) > eq_tol).astype(int)


def do_lists_have_same_elements(a: Iterable, b: Iterable) -> bool:
    """Whether two lists only contains the same elements (not necessarily in the same quantities)
    Args:
        a (Iterable): the first list
        b (Iterable): the second list
    Returns:
        bool: Whether both lists contain the same elements
    """
    return collections.Counter(a) == collections.Counter(b)
