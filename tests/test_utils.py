import pytest
from unittest.mock import patch
import numpy as np
from os import PathLike
import io

import fauvqe.utils as utils

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
         (  utils.direct_sum,
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
        (np.matmul,([[3, 0], [0, 2]], [[0, 2], [2, 0]], [[1, 1], [1, 1]]), [[6, 6], [4, 4]])
    ],
)
def test_chained_matrix_multiplication(multiplication_rule,l, correct):
    assert (np.array(utils.chained_matrix_multiplication(multiplication_rule,*l)) == np.array(correct)).all()


@pytest.mark.parametrize(
    "l,correct",
    [
        (
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
            [np.eye(2), np.ones((2, 2)), np.ones((2, 2))],
            np.kron(np.eye(2), np.ones((4, 4))),
        ),

    ],
)
def test_pi_kron(l, correct):
    assert (np.array(utils.pi_kron(*l)) == np.array(correct)).all()


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
    assert (utils.direct_sum(np.array(a), np.array(b)) == np.array(correct)).all()


@pytest.mark.parametrize(
    "l,correct",
    [
        (
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
    ],
)
def test_pi_direct_sum(l, correct):
    assert (utils.pi_direct_sum(*l) == correct).all()


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
    assert list(utils.flatten(a)) == correct


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
        utils.alternating_indices_to_sectors(np.array(M), even_first, axis) == correct
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
        utils.sectors_to_alternating_indices(np.array(M), even_first, axis)
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
    assert (np.array(utils.flip_cross_rows(M, flip_odd) == correct)).all()


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
    assert (np.array(utils.flip_cross_cols(M, flip_odd) == correct)).all()


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
    assert (utils.flip_cross(np.array(M), rc, flip_odd) == correct).all()


def test_flip_cross_error():
    with pytest.raises(ValueError):
        utils.flip_cross(M=[1, 2], rc="u")


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
    assert utils.hamming_weight(i) == correct


def test_hamming_weight_error():
    with pytest.raises(TypeError):
        utils.hamming_weight("lol")
        utils.hamming_weight({"a": 2})


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
    assert utils.index_bits(a, ones) == correct


@pytest.mark.parametrize(
    "l,correct",
    [(([[3, 0], [0, 2]], [[0, 2], [2, 0]], [[1, 1], [1, 1]]), [[6, 6], [4, 4]])],
)
def test_pi_matmul(l, correct):
    assert np.array(utils.pi_matmul(*l) == correct).all()


@pytest.mark.parametrize(
    "M,correct,eq_tol",
    [
        ([[2, 2], [2, 2]], [[1, 1], [1, 1]], 1e-1),
        ([[2, 2], [2, 2]], [[0, 0], [0, 0]], 1e2),
        ([[2, 2e-16], [2e-16, 2]], [[1, 0], [0, 1]], 1e-1),
        ([[2, -2], [-2, -2e-1]], [[1, 1], [1, 1]], 1e-1),
    ],
)
def test_get_non_zero(M, correct, eq_tol):
    assert np.array(utils.get_non_zero(M=M, eq_tol=eq_tol) == correct).all()


@pytest.mark.parametrize(
    "M1,M2,correct",
    [
        ([1, 2, 3, 4], [8, 9, 10, 11], [1, 8, 2, 9, 3, 10, 4, 11]),
        ([1, 2, 3, 4, 5], [8, 9, 10, 11], [1, 8, 2, 9, 3, 10, 4, 11, 5]),
        ([1, 2], [8, 9, 10, 11], [1, 8, 2, 9, 10, 11]),
    ],
)
def test_interweave(M1, M2, correct):
    assert np.array(utils.interweave(M1, M2) == correct).all()


@pytest.mark.parametrize(
    "M1,M2,correct",
    [
        ([1, 2, 3, 4], [8, 9, 10, 11], False),
        ([1, 2, 3, 4], [3, 2, 1, 4], True),
        (["a", "b"], ["b", "a"], True),
        ([1, 2, 3, 4], [3, 3, 3, 1, 2, 4], False),
    ],
)
def test_lists_have_same_elements(M1, M2, correct):
    assert utils.lists_have_same_elements(M1, M2) == correct


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
        utils.arg_alternating_indices_to_sectors(indices=indices, N=N) == correct
    ).all()


def test_arg_alternating_indices_to_sectors_error():
    with pytest.raises(ValueError):
        utils.arg_alternating_indices_to_sectors(
            indices=[1, 2],
            N=[
                2,
            ],
        )
    with pytest.raises(TypeError):
        utils.arg_alternating_indices_to_sectors(indices=[1, 2], N="a")


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
    assert utils.arg_flip_cross_row(x=x, y=y, dimy=dimy, flip_odd=flip_odd) == correct


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
        utils.arg_flip_cross_row(x, y, dimy)


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
    assert utils.grid_to_linear(x, y, dimx, dimy, horizontal) == correct


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
    assert utils.linear_to_grid(n, dimx, dimy, horizontal) == correct


@pytest.mark.parametrize(
    "v,correct",
    [((1, 1), np.array((1 / np.sqrt(2), 1 / np.sqrt(2)))), ((1, 0), (1, 0))],
)
def test_normalize(v, correct):
    assert (utils.normalize(v) == correct).all()


@pytest.mark.parametrize(
    "l,i,correct",
    [
        ([2, 4, 6, 8], 2, 4),
        ([1, 3, 7, 9], 2, 0),
    ],
)
def test_sum_divisible(l, i, correct):
    assert utils.sum_divisible(l, i) == correct


@pytest.mark.parametrize(
    "l,correct",
    [
        ([10, 20], 2),
        ([11, 21, 22], 1),
    ],
)
def test_sum_even(l, correct):
    assert utils.sum_even(l) == correct


@pytest.mark.parametrize(
    "l,correct",
    [
        ([10, 20], 0),
        ([11, 21, 22], 2),
    ],
)
def test_sum_odd(l, correct):
    assert utils.sum_odd(l) == correct


@pytest.mark.parametrize(
    "fpath",
    [
        "/obviously/fake/dir/file.txt",
    ],
)
def test_ensure_fpath(fpath):
    with patch("os.makedirs") as mock_makedir:
        mock_makedir.side_effect = lambda x: True
        utils.ensure_fpath(fpath)
        mock_makedir.assert_called_once()


@pytest.mark.parametrize(
    "s,token,correct",
    [
        ("s a#x/o.p_h*o+n)e", "-", "s-a-x-o-p-h-o-n-e"),
    ],
)
def test_normalize_str(s, token, correct):
    assert utils.normalize_str(s, token) == correct


@pytest.mark.parametrize(
    "s,correct",
    [("x", "X"), ("bonjour", "Bonjour")],
)
def test_cap_first(s, correct):
    assert utils.cap_first(s) == correct


@pytest.mark.parametrize(
    "a,b,correct",
    [
        ((1, 0), (1, 0), 1),
        ((0, 1), (1, 0), 0),
        (((2, 2), (2, 2)), (1, 0), np.sqrt(2)),
        ((1, 0), ((2, 2), (2, 2)), np.sqrt(2)),
        (((2, 2), (2, 2)), ((2, 2), (2, 2)), 4),
    ],
)
def test_fidelity(a, b, correct):
    assert np.isclose(utils.fidelity(a, b), correct)
    assert np.isclose(utils.fidelity(a, b), utils.fidelity(b, a))
    assert np.isclose(utils.infidelity(a, b), 1 - correct)


@pytest.mark.parametrize(
    "a,b",
    [
        ((1, 2), (1, 2, 3)),
        ((1, 2, 3), ((1, 2, 3), (1, 2, 3, 3))),
        ((1, 2, 3, 4), ((1, 2, 3), (1, 2, 3))),
        (
            (
                1,
                2,
            ),
            (((1, 2), (1, 2)), ((1, 2), (1, 2))),
        ),
    ],
)
def test_error_fidelity(a, b):
    with pytest.raises(ValueError):
        utils.fidelity(a, b)


@pytest.mark.parametrize(
    "data, dirname, fname, randname, date",
    [
        ({"x": 1}, None, None, False, False),
        ({"x": 1}, "dir/", "file", True, True),
    ],
)
def test_save_to_json(data, dirname, fname, randname, date):
    with patch("io.open") as mock_open:
        with patch("io.TextIOWrapper") as mock_iowrapper:
            with patch("fauvqe.utils.ensure_fpath") as mock_ensure_fpath:
                utils.save_to_json(
                    data=data,
                    dirname=dirname,
                    fname=fname,
                    randname=randname,
                    date=date,
                )
                mock_open.assert_called()
                mock_ensure_fpath.assert_called()


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
    neighbours = utils.grid_neighbour_list(
        i, shape, neighbour_order, periodic, diagonal, origin
    )
    print(neighbours)
    assert utils.lists_have_same_elements(neighbours, correct)


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
            utils.default_value_handler(shape, value).shape == np.array(shape)
        ).all()
    elif value in ["zeros", "ones"] or isinstance(value, float):
        assert (utils.default_value_handler(shape, value) == correct).all()
    else:
        with pytest.raises(ValueError):
            utils.default_value_handler(shape, value)


@pytest.mark.parametrize(
    "arr,indices,correct",
    [("abcde", [1, 2, 5, 6, 7], "bcabc"), ([10, 20, 30], [100, 200, 0], [20, 30, 10])],
)
def test_wrapping_slice(arr, indices, correct):
    assert utils.wrapping_slice(arr, indices) == correct


@pytest.mark.parametrize(
    "indices, Nqubits, correct",
    [
        ([1, 2], 3, np.array([0, 0, 0, 0, 0, 0, 1, 0])),
        ([0, 1], 2, np.array([0, 0, 0, 1])),
    ],
)
def test_jw_computational_wf(indices, Nqubits, correct):
    assert (utils.jw_computational_wf(indices, Nqubits) == correct).all()
