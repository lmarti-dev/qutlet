import pytest
import numpy as np

import fauvqe.utils as utils

@pytest.mark.parametrize(
    "l,correct",
    [
        ([[[1,2,3],[0,0,0],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]],
            [[1,1,1,2,2,2,3,3,3],
            [1,1,1,2,2,2,3,3,3],
            [1,1,1,2,2,2,3,3,3],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1]]
            
        ),
    ]
)
def test_pi_kron(l,correct):
    assert (utils.pi_kron(*l) == np.array(correct)).all()
@pytest.mark.parametrize(
    "a,b,correct",
    [
        ([[1,2,3],[0,0,0],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],
           [[1,2,3,0,0,0],
           [0,0,0,0,0,0],
           [1,1,1,0,0,0],
           [0,0,0,1,1,1],
           [0,0,0,1,1,1],
           [0,0,0,1,1,1]]  
        ),
    ]
)
def test_direct_sum(a,b,correct):
    assert (utils.direct_sum(np.array(a),np.array(b)) == np.array(correct)).all()
@pytest.mark.parametrize(
    "l,correct",
    [
        ([np.array([[1,2,3],[0,0,0],[1,1,1]]),
            np.array([[1,1,1],[1,1,1],[1,1,1]]),
            np.array([[2,2,2],[2,2,2],[3,3,3]])
        ],
           np.array(
               [[1,2,3,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [1,1,1,0,0,0,0,0,0],
                [0,0,0,1,1,1,0,0,0],
                [0,0,0,1,1,1,0,0,0],
                [0,0,0,1,1,1,0,0,0],
                [0,0,0,0,0,0,2,2,2],
                [0,0,0,0,0,0,2,2,2],
                [0,0,0,0,0,0,3,3,3]]
            )  
        ),
    ]
)
def test_pi_direct_sum(l,correct):
    assert (utils.pi_direct_sum(*l) == correct).all()
@pytest.mark.parametrize(
    "a,correct",
    [
        ([[1,2,3],[],[4,[5,[6,7]],8,9]],[1,2,3,4,5,6,7,8,9]),
        ([[1,2,3],[4,[5,[6]]],7,8,[9]],[1,2,3,4,5,6,7,8,9]),
        ([[[],[1],[2,3]],[],[4,[5,6,7],8,9]],[1,2,3,4,5,6,7,8,9]),
        ([1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]),
    ]
)
def test_flatten(a,correct):
    assert list(utils.flatten(a)) == correct

@pytest.mark.parametrize(
    "M,correct,even_first",
    [
     ([0,1,2,3,4,5],[0,2,4,1,3,5],True), 
     ([[0,1,2,3,4,5],[0,1,2,3,4,5]],[[0,2,4,1,3,5],[0,2,4,1,3,5]],True), 
     ([[0,1,2,3,4,5],[6,7,8,9,10,11],[12,13,14,15,16,17],[18,19,20,21,22,23]],
            [[0,2,4,1,3,5],[12,14,16,13,15,17],[6,8,10,7,9,11],[18,20,22,19,21,23]],True), 
    ]
)

def test_alternating_indices_to_sectors(M,correct,even_first):
    assert (utils.alternating_indices_to_sectors(np.array(M),even_first) == correct).all()

@pytest.mark.parametrize(
    "M,correct,even_first",
    [
     ([0,2,4,1,3,5],[0,1,2,3,4,5],True), 
     ([[0,2,4,1,3,5],[0,2,4,1,3,5]],[[0,1,2,3,4,5],[0,1,2,3,4,5]],True), 
     ([[0,2,4,1,3,5],[12,14,16,13,15,17],[6,8,10,7,9,11],[18,20,22,19,21,23]],
        [[0,1,2,3,4,5],[6,7,8,9,10,11],[12,13,14,15,16,17],[18,19,20,21,22,23]],True), 
    ]
)

def test_sectors_to_alternating_indices(M,correct,even_first):
    assert (utils.sectors_to_alternating_indices(np.array(M),even_first) == correct).all()


@pytest.mark.parametrize(
    "M,correct,flip_odd",
    [
     (
        [[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5]]
        ,[[0,1,2,3,4,5],[5,4,3,2,1,0],[0,1,2,3,4,5]]
        ,True
     ),
    (
        [[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5]]
        ,[[0,1,2,3,4,5],[5,4,3,2,1,0],[0,1,2,3,4,5],[5,4,3,2,1,0]]
        ,True
    ),
    (
        [[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5]]
        ,[[5,4,3,2,1,0],[0,1,2,3,4,5],[5,4,3,2,1,0],[0,1,2,3,4,5]]
        ,False
    ),

    ]
)
def test_flip_cross_rows(M,correct,flip_odd):
    assert (utils.flip_cross_rows(np.array(M),flip_odd)==correct).all()
@pytest.mark.parametrize(
    "i,correct",
    [
     (bin(24),2),
     (bin(-1),1),
     (bin(300),4),
     (bin(3),2),
     (199,5),
     (19,3),
     (bin(0),0),
     (0,0),
    ]
)
def test_hamming_weight(i,correct):
    assert utils.hamming_weight(i)==correct

@pytest.mark.parametrize(
    "a,correct",
    [
     (bin(300),[0,3,5,6]),
     (bin(399),[0,1,5,6,7,8]),
     (bin(1),[0]),
     (bin(0),[]),
     (bin(-29),[0,1,2,4]),
    ]
)
def test_indexbits(a,correct):
    assert utils.index_bits(a)==correct

@pytest.mark.parametrize(
    "l,correct",
    [
        (([[3,0],[0,2]],[[0,2],[2,0]],[[1,1],[1,1]]),[[6,6],[4,4]])
    ])
def test_pi_matmul(l,correct):
    assert np.array(utils.pi_matmul(*l) == correct).all()

@pytest.mark.parametrize(
    "M,correct,eq_tol",
    [
        ([[2,2],[2,2]],[[1,1],[1,1]],1E-1),
        ([[2,2],[2,2]],[[0,0],[0,0]],1E2),
        ([[2,2E-16],[2E-16,2]],[[1,0],[0,1]],1E-1),
        ([[2,-2],[-2,-2E-1]],[[1,1],[1,1]],1E-1),
    ])
def test_get_non_zero(M,correct,eq_tol):
    assert np.array(utils.get_non_zero(M=M,eq_tol=eq_tol) == correct).all()

@pytest.mark.parametrize(
    "M1,M2,correct",
    [
        ([1,2,3,4],[8,9,10,11],[1,8,2,9,3,10,4,11]),
        ([1,2,3,4,5],[8,9,10,11],[1,8,2,9,3,10,4,11,5]),
        ([1,2],[8,9,10,11],[1,8,2,9,10,11]),
    ])
def test_interweave(M1,M2,correct):
    assert np.array(utils.interweave(M1,M2) == correct).all()

@pytest.mark.parametrize(
    "M1,M2,correct",
    [
        ([1,2,3,4],[8,9,10,11],False),
        ([1,2,3,4],[3,2,1,4],True),
        (["a","b"],["b","a"],True),
        ([1,2,3,4],[3,3,3,1,2,4],False),
    ])
def test_lists_have_same_elements(M1,M2,correct):
    assert utils.lists_have_same_elements(M1,M2) == correct

@pytest.mark.parametrize(
    "M,correct,eq_tol",
    [
        ([1,2,3,4],[0,0,3,4],2.5),
        ([-1E-12,1E-12,3,4],[0,0,3,4],1E-10),
        ([1,2,3,4],[0,0,3,4],3),
    ])
def test_round_small_to_zero(M,correct,eq_tol):
    assert utils.round_small_to_zero(l=M,eq_tol=eq_tol) == correct


@pytest.mark.parametrize(
    "M,correct",
    [
        ([[1j,2j],[3j,4j]],[[-1j, -3j],[-2j, -4j]]),
        ([[1,2],[3,4]],[[1, 3],[2, 4]]),
        ([[1j,2],[3,4j]],[[-1j, 3],[2, -4j]]),
        
    ])
def test_unitary_transpose(M,correct):
    assert np.array(utils.unitary_transpose(np.array(M)) == correct).all()

@pytest.mark.parametrize(
    "indices,correct,N",
    [
        ((0,1,2,3,4,5,6,7),(0,4,1,5,2,6,3,7),8),
        ((3,2),(6,1),10),
    ])
def test_arg_alternating_indices_to_sectors(indices,correct,N):
    assert np.array(utils.arg_alternating_indices_to_sectors(indices=indices,N=N) == correct).all()
@pytest.mark.parametrize(
    "x,y,dimy,correct,flip_odd",
    [
        (2,1,4,(2,1),True),
        (1,1,4,(1,2),True),
        (1,0,4,(1,3),True),
        (12,1,4,(12,2),False),
        (11,1,4,(11,1),False),
    ])
def test_arg_flip_cross_row(x,y,dimy,correct,flip_odd):
    assert utils.arg_flip_cross_row(x=x,y=y,dimy=dimy,flip_odd=flip_odd) == correct
@pytest.mark.parametrize(
    "x,y,dimy",
    [
        (-2,1,4),
        (2,-1,4),
        (2,1,-4),
        (2,10,4),
    ])
def test_arg_flip_cross_row_error(x,y,dimy):
    with pytest.raises(ValueError):
        utils.arg_flip_cross_row(x,y,dimy)

@pytest.mark.parametrize(
    "x,y,dimx,dimy,correct",
    [
        (1,2,2,4,6),
        (0,0,2,4,0),
        (0,3,2,4,3),
        (2,1,4,2,5),
        (3,1,4,2,7),
    ])
def test_grid_to_linear(x,y,dimx,dimy,correct):
    assert utils.grid_to_linear(x,y,dimx,dimy) == correct

@pytest.mark.parametrize(
    "n,dimx,dimy,correct",
    [
        (6,2,4,(1,2)),
        (6,4,2,(3,0)),
        (0,10,10,(0,0)),
        (6,2,4,(1,2)),
    ])
def test_linear_to_grid(n,dimx,dimy,correct):
    assert utils.linear_to_grid(n,dimx,dimy) == correct

