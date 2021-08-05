import pytest
import numpy as np
from scipy.stats import unitary_group

from fauvqe import MatrixCost, Ising
    
@pytest.mark.parametrize(
    "n",
    [
        (
            2
        ), 
        (
            [2, 2]
        ),  
        (
            4
        ), 
        (
            [4, 4]
        ),
        (
            8
        ), 
        (
            [8, 8]
        ), 
        (
            16
        ), 
        (
            [16, 16]
        ), 
    ],
)
def test_evaluate(n):
    np.set_printoptions(precision=16)
    #Generate Ising Object as MockAbstractModel
    mockmodel = Ising("GridQubit", [1, 1], np.ones((0, 1)), np.ones((1, 1)), np.ones((1, 1)))

    #Generate random state vector or unitary
    if isinstance(n,int):
        rand_matrix = np.random.rand(n) + 1j*np.random.rand(n)
        rand_matrix /= np.linalg.norm(rand_matrix)
    else:
        rand_matrix = unitary_group.rvs(n[0])

    #Generate MatrixCost object and test by evaluating it by comparing 
    #rand_matrix with itself
    objective = MatrixCost(mockmodel, rand_matrix)
    assert objective.evaluate(rand_matrix) < 1e-15
    
@pytest.mark.parametrize(
    "t",
    [
        (0.1), #(15), (-0.01)
    ],
)
def test_simulate(t):
    assert False

#############################################################
#                                                           #
#                    Assert tests                           #
#                                                           #
#############################################################