import numpy as np
import pytest

import fauvqe.utilities.testing
import fauvqe.utilities.generic


@pytest.mark.parametrize(
    "M,correct,eq_tol",
    [
        ([[2, 2], [2, 2]], [[1, 1], [1, 1]], 1e-1),
        ([[2, 2], [2, 2]], [[0, 0], [0, 0]], 1e2),
        ([[2, 2e-16], [2e-16, 2]], [[1, 0], [0, 1]], 1e-1),
        ([[2, -2], [-2, -2e-1]], [[1, 1], [1, 1]], 1e-1),
    ],
)
def test_non_zero_matrix(M, correct, eq_tol):
    assert np.array(
        fauvqe.utilities.testing.non_zero_matrix(M=M, eq_tol=eq_tol) == correct
    ).all()


@pytest.mark.parametrize(
    "M1,M2,correct",
    [
        ([1, 2, 3, 4], [8, 9, 10, 11], False),
        ([1, 2, 3, 4], [3, 2, 1, 4], True),
        (["a", "b"], ["b", "a"], True),
        ([1, 2, 3, 4], [3, 3, 3, 1, 2, 4], False),
    ],
)
def test_do_lists_have_same_elements(M1, M2, correct):
    assert fauvqe.utilities.testing.do_lists_have_same_elements(M1, M2) == correct
