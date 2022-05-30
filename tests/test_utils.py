import pytest
import numpy as np

import fauvqe.utils as ut

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
    assert (ut.pi_kron(*l) == np.array(correct)).all()
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
    assert (ut.direct_sum(np.array(a),np.array(b)) == np.array(correct)).all()
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
    assert (ut.pi_direct_sum(*l) == correct).all()
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
    assert list(ut.flatten(a)) == correct

"""
todo
def test_pi_matmul(l,correct):
    assert ut.pi_matmul(*l) == correct

def test_print_non_zero(M,correct):
    assert ut.print_non_zero(M) == correct
"""

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
    assert (ut.alternating_indices_to_sectors(np.array(M),even_first) == correct).all()

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
    assert (ut.sectors_to_alternating_indices(np.array(M),even_first) == correct).all()


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
    assert (ut.flip_cross_rows(np.array(M),flip_odd)==correct).all()
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
    assert ut.hamming_weight(i)==correct

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
    assert ut.index_bits(a)==correct
