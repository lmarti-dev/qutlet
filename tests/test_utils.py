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

