import utilities.generic


import numpy as np
import pytest

import fauvqe.utilities.testing
import fauvqe.utilities.generic


@pytest.mark.parametrize(
    "multiplication_rule,l,correct",
    [
        (
            np.kron,
            [[[1, 2, 3], [0, 0, 0], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
            [
                [1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 1, 1, 2, 2, 2, 3, 3, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
        ),
        (
            np.kron,
            [np.eye(2), np.ones((2, 2)), np.ones((2, 2))],
            np.kron(np.eye(2), np.ones((4, 4))),
        ),
        (
            fauvqe.utilities.generic.direct_sum,
            [
                np.array([[1, 2, 3], [0, 0, 0], [1, 1, 1]]),
                np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                np.array([[2, 2, 2], [2, 2, 2], [3, 3, 3]]),
            ],
            np.array(
                [
                    [1, 2, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 2, 2, 2],
                    [0, 0, 0, 0, 0, 0, 2, 2, 2],
                    [0, 0, 0, 0, 0, 0, 3, 3, 3],
                ]
            ),
        ),
        (
            np.matmul,
            ([[3, 0], [0, 2]], [[0, 2], [2, 0]], [[1, 1], [1, 1]]),
            [[6, 6], [4, 4]],
        ),
    ],
)
def test_chained_matrix_multiplication(multiplication_rule, l, correct):
    assert (
        np.array(
            fauvqe.utilities.generic.chained_matrix_multiplication(
                multiplication_rule, *l
            )
        )
        == np.array(correct)
    ).all()


@pytest.mark.parametrize(
    "a,correct",
    [
        ([[1, 2, 3], [], [4, [5, [6, 7]], 8, 9]], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ([[1, 2, 3], [4, [5, [6]]], 7, 8, [9]], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ([[[], [1], [2, 3]], [], [4, [5, 6, 7], 8, 9]], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ],
)
def test_flatten(a, correct):
    assert list(fauvqe.utilities.generic.flatten(a)) == correct


@pytest.mark.parametrize(
    "a,b,correct",
    [
        (
            [[1, 2, 3], [0, 0, 0], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [1, 2, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
            ],
        ),
    ],
)
def test_direct_sum(a, b, correct):
    assert (
        fauvqe.utilities.generic.direct_sum(np.array(a), np.array(b))
        == np.array(correct)
    ).all()


@pytest.mark.parametrize(
    "M,correct,even_first,axis",
    [
        ([0, 1, 2, 3, 4, 5], [0, 2, 4, 1, 3, 5], True, None),
        ([[0, 1], [2, 3]], [[2, 3], [0, 1]], False, 0),
        (
            [0, 1, 2, 3, 4, 5],
            [
                1,
                3,
                5,
                0,
                2,
                4,
            ],
            False,
            None,
        ),
        (
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            [[0, 2, 4, 1, 3, 5], [0, 2, 4, 1, 3, 5]],
            True,
            None,
        ),
        (
            [
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
                [12, 13, 14, 15, 16, 17],
                [18, 19, 20, 21, 22, 23],
            ],
            [
                [0, 2, 4, 1, 3, 5],
                [12, 14, 16, 13, 15, 17],
                [6, 8, 10, 7, 9, 11],
                [18, 20, 22, 19, 21, 23],
            ],
            True,
            None,
        ),
    ],
)
def test_alternating_indices_to_sectors(M, correct, even_first, axis):
    assert (
        fauvqe.utilities.generic.alternating_indices_to_sectors(
            np.array(M), even_first, axis
        )
        == correct
    ).all()


@pytest.mark.parametrize(
    "M,correct,even_first,axis",
    [
        ([0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5], True, 0),
        ([0, 2, 4, 1, 3, 5], [1, 0, 3, 2, 5, 4], False, None),
        (
            [[0, 2, 4, 1, 3, 5], [0, 2, 4, 1, 3, 5]],
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            True,
            (0, 1),
        ),
        (
            [
                [0, 2, 4, 1, 3, 5],
                [12, 14, 16, 13, 15, 17],
                [6, 8, 10, 7, 9, 11],
                [18, 20, 22, 19, 21, 23],
            ],
            [
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
                [12, 13, 14, 15, 16, 17],
                [18, 19, 20, 21, 22, 23],
            ],
            True,
            (0, 1),
        ),
    ],
)
def test_sectors_to_alternating_indices(M, correct, even_first, axis):
    assert (
        fauvqe.utilities.generic.sectors_to_alternating_indices(
            np.array(M), even_first, axis
        )
        == np.array(correct)
    ).all()


@pytest.mark.parametrize(
    "M,correct,flip_odd",
    [
        (
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]],
            True,
        ),
        (
            [
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
            ],
            [
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
            ],
            True,
        ),
        (
            np.array(
                [
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                ]
            ),
            np.array(
                [
                    [5, 4, 3, 2, 1, 0],
                    [0, 1, 2, 3, 4, 5],
                    [5, 4, 3, 2, 1, 0],
                    [0, 1, 2, 3, 4, 5],
                ]
            ),
            False,
        ),
    ],
)
def test_flip_cross_rows(M, correct, flip_odd):
    assert (
        np.array(fauvqe.utilities.generic.flip_cross_rows(M, flip_odd) == correct)
    ).all()


@pytest.mark.parametrize(
    "M,correct,rc,flip_odd",
    [
        (
            [[1, 2], [3, 4]],
            [[1, 2], [4, 3]],
            "r",
            True,
        ),
        (
            [[1, 2], [3, 4]],
            [[3, 2], [1, 4]],
            "c",
            False,
        ),
    ],
)
def test_flip_cross(M, correct, rc, flip_odd):
    assert (
        fauvqe.utilities.generic.flip_cross(np.array(M), rc, flip_odd) == correct
    ).all()


@pytest.mark.parametrize(
    "M1,M2,correct",
    [
        ([1, 2, 3, 4], [8, 9, 10, 11], [1, 8, 2, 9, 3, 10, 4, 11]),
        ([1, 2, 3, 4, 5], [8, 9, 10, 11], [1, 8, 2, 9, 3, 10, 4, 11, 5]),
        ([1, 2], [8, 9, 10, 11], [1, 8, 2, 9, 10, 11]),
    ],
)
def test_interweave(M1, M2, correct):
    assert np.array(fauvqe.utilities.generic.interweave(M1, M2) == correct).all()


@pytest.mark.parametrize(
    "indices,correct,N",
    [
        ((0, 1, 2, 3, 4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 3, 7), 8),
        ((3, 2), (6, 1), 10),
        ((0, 2), (0, 1), (2, 4)),
    ],
)
def test_arg_alternating_indices_to_sectors(indices, correct, N):
    assert np.array(
        fauvqe.utilities.generic.arg_alternating_indices_to_sectors(
            indices=indices, N=N
        )
        == correct
    ).all()


def test_arg_alternating_indices_to_sectors_error():
    with pytest.raises(ValueError):
        fauvqe.utilities.generic.arg_alternating_indices_to_sectors(
            indices=[1, 2],
            N=[
                2,
            ],
        )
    with pytest.raises(TypeError):
        fauvqe.utilities.generic.arg_alternating_indices_to_sectors(
            indices=[1, 2], N="a"
        )


@pytest.mark.parametrize(
    "x,y,dimy,correct,flip_odd",
    [
        (2, 1, 4, (2, 1), True),
        (1, 1, 4, (1, 2), True),
        (1, 0, 4, (1, 3), True),
        (12, 1, 4, (12, 2), False),
        (11, 1, 4, (11, 1), False),
    ],
)
def test_arg_flip_cross_row(x, y, dimy, correct, flip_odd):
    assert (
        fauvqe.utilities.generic.arg_flip_cross_row(
            x=x, y=y, dimy=dimy, flip_odd=flip_odd
        )
        == correct
    )


@pytest.mark.parametrize(
    "x,y,dimy",
    [
        (-2, 1, 4),
        (2, -1, 4),
        (2, 1, -4),
        (2, 10, 4),
    ],
)
def test_arg_flip_cross_row_error(x, y, dimy):
    with pytest.raises(ValueError):
        fauvqe.utilities.generic.arg_flip_cross_row(x, y, dimy)


@pytest.mark.parametrize(
    "x,y,dimx,dimy,correct,horizontal",
    [
        (1, 2, 2, 4, 6, True),
        (0, 0, 2, 4, 0, True),
        (0, 3, 2, 4, 3, True),
        (2, 1, 4, 2, 5, True),
        (3, 1, 4, 2, 7, True),
        (4, 1, 4, 2, 8, False),
    ],
)
def test_grid_to_linear(x, y, dimx, dimy, correct, horizontal):
    assert (
        fauvqe.utilities.generic.grid_to_linear(x, y, dimx, dimy, horizontal) == correct
    )


@pytest.mark.parametrize(
    "n,dimx,dimy,correct,horizontal",
    [
        (6, 2, 4, (1, 2), True),
        (6, 4, 2, (3, 0), True),
        (0, 10, 10, (0, 0), True),
        (6, 2, 4, (1, 2), True),
        (6, 4, 4, (2, 1), False),
    ],
)
def test_linear_to_grid(n, dimx, dimy, correct, horizontal):
    assert fauvqe.utilities.generic.linear_to_grid(n, dimx, dimy, horizontal) == correct


@pytest.mark.parametrize(
    "v,correct",
    [((1, 1), np.array((1 / np.sqrt(2), 1 / np.sqrt(2)))), ((1, 0), (1, 0))],
)
def test_normalize_vec(v, correct):
    assert (fauvqe.utilities.generic.normalize_vec(v) == correct).all()


@pytest.mark.parametrize(
    "l,i,correct",
    [
        ([2, 4, 6, 8], 2, 4),
        ([1, 3, 7, 9], 2, 0),
    ],
)
def test_sum_divisible(l, i, correct):
    assert fauvqe.utilities.generic.sum_divisible(l, i) == correct


@pytest.mark.parametrize(
    "l,correct",
    [
        ([10, 20], 2),
        ([11, 21, 22], 1),
    ],
)
def test_sum_even(l, correct):
    assert fauvqe.utilities.generic.sum_even(l) == correct


@pytest.mark.parametrize(
    "l,correct",
    [
        ([10, 20], 0),
        ([11, 21, 22], 2),
    ],
)
def test_sum_odd(l, correct):
    assert fauvqe.utilities.generic.sum_odd(l) == correct


@pytest.mark.parametrize(
    "s,token,correct",
    [
        ("s a#x/o.p_h*o+n)e", "-", "s-a-x-o-p-h-o-n-e"),
    ],
)
def test_replace_non_alpha(s, token, correct):
    assert fauvqe.utilities.generic.replace_non_alpha(s, token) == correct


@pytest.mark.parametrize(
    "a,correct,ones",
    [
        (bin(300), [0, 3, 5, 6], True),
        (bin(399), [0, 1, 5, 6, 7, 8], True),
        (bin(1), [0], True),
        (bin(0), [], True),
        (bin(-29), [0, 1, 2, 4], True),
        (10, [1, 3], False),
    ],
)
def test_index_bits(a, correct, ones):
    assert fauvqe.utilities.generic.index_bits(a, ones) == correct


@pytest.mark.parametrize(
    "i,shape,neighbour_order,periodic,diagonal,origin,correct",
    [
        (3, (4, 4), 1, True, True, "center", (0, 2, 3, 4, 6, 7, 12, 14, 15)),
        (2, (3, 6), 2, False, False, "topleft", (2, 3, 4, 8, 14)),
    ],
)
def test_grid_neighbour_list(
    i, shape, neighbour_order, periodic, diagonal, origin, correct
):
    neighbours = fauvqe.utilities.generic.grid_neighbour_list(
        i, shape, neighbour_order, periodic, diagonal, origin
    )
    print(neighbours)
    assert fauvqe.utilities.testing.do_lists_have_same_elements(neighbours, correct)


@pytest.mark.parametrize(
    "i,correct",
    [
        (bin(24), 2),
        (bin(-1), 1),
        (bin(300), 4),
        (bin(3), 2),
        (199, 5),
        (19, 3),
        (bin(0), 0),
        (0, 0),
    ],
)
def test_hamming_weight(i, correct):
    assert fauvqe.utilities.generic.hamming_weight(i) == correct


def test_hamming_weight_error():
    with pytest.raises(TypeError):
        fauvqe.utilities.generic.hamming_weight("lol")
        fauvqe.utilities.generic.hamming_weight({"a": 2})


@pytest.mark.parametrize(
    "shape,value,correct",
    [
        ((3, 3), 3.3, np.full(shape=(3, 3), fill_value=3.3)),
        ((2, 2), "zeros", np.zeros(shape=(2, 2))),
        ((1, 1), "ones", np.ones(shape=(1, 1))),
        ((3, 3), "random", np.random.rand(3, 3)),
        ((10, 10), "zoink", None),
    ],
)
def test_default_value_handler(shape, value, correct):
    if value == "random":
        assert (
            fauvqe.utilities.generic.default_value_handler(shape, value).shape
            == np.array(shape)
        ).all()
    elif value in ["zeros", "ones"] or isinstance(value, float):
        assert (
            fauvqe.utilities.generic.default_value_handler(shape, value) == correct
        ).all()
    else:
        with pytest.raises(ValueError):
            fauvqe.utilities.generic.default_value_handler(shape, value)


@pytest.mark.parametrize(
    "arr,indices,correct",
    [("abcde", [1, 2, 5, 6, 7], "bcabc"), ([10, 20, 30], [100, 200, 0], [20, 30, 10])],
)
def test_wrapping_slice(arr, indices, correct):
    assert fauvqe.utilities.generic.wrapping_slice(arr, indices) == correct


def test_flip_cross_error():
    with pytest.raises(ValueError):
        fauvqe.utilities.generic.flip_cross(M=[1, 2], rc="u")


@pytest.mark.parametrize(
    "M,correct,flip_odd",
    [
        (
            [[1, 2], [3, 4]],
            [[1, 4], [3, 2]],
            True,
        ),
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[3, 2], [1, 4]]),
            False,
        ),
    ],
)
def test_flip_cross_cols(M, correct, flip_odd):
    assert (
        np.array(fauvqe.utilities.generic.flip_cross_cols(M, flip_odd) == correct)
    ).all()
