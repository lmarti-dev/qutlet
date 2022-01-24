"""
    Test to be added:
        -test periodic boundaries
        -1D TFIM analytic result is inconsistent with QAOA energy:
            E_QAOA < E_analytic seems clearly wrong.
        -Need better TFIM tests as <Z>_X = <X>_Z = 0; 
            hence using Z/X eigenstates is generally a bad idea
"""
# external imports
import pytest
import numpy as np
import cirq
from scipy.linalg import expm
import sympy

# internal imports
from fauvqe import TwoLevel, ExpectationValue
from tests.test_isings import IsingTester

def test__eq__():
    n = [1,3]; boundaries = [1, 0]
    model = TwoLevel("GridQubit", n, np.ones((n[0], n[1])))
    model.set_circuit("qaoa")
    
    model2 = TwoLevel("GridQubit", n, np.ones((n[0], n[1])))
    model2.set_circuit("qaoa")
    
    assert (model == model2)

    model.set_Ut()
    assert model != model2 


@pytest.mark.parametrize(
    "qubittype, n, h",
    [
        #############################################################
        #                   2 qubit tests                           #
        #############################################################
        (
            "GridQubit",
            [1, 2],
            np.ones((1, 2)) / 3,
        ),
        #############################################################
        #                   4 qubit tests                           #
        #############################################################
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2)) / 3,
        ),
    ],
)
def test_copy(qubittype, n, h):
    model = TwoLevel(qubittype, n, h)
    model.set_circuit("qaoa")
    model2 = model.copy()

    #Test whether the objects are the same
    assert( model == model2 )

    #But there ID is different
    assert( model is not model2 )


@pytest.mark.parametrize(
    "n, h, index, sol",
    [
        ([1, 2],np.ones((1, 2)), 0, 1),
        ([1, 2],np.ones((1, 2)), 1, 0),
        ([1, 2],np.ones((1, 2)), 3, -1),
        ([2, 2], np.ones((2, 2)), 0, 1),
        ([2, 2], np.ones((2, 2)), 12, 0),
        ([2, 2], np.ones((2, 2)), 11, -0.5),
    ],
)
def test_energy(n, h, index, sol):
    model = TwoLevel("GridQubit", n, h)
    energy_obs = ExpectationValue(model)
    
    wf = np.zeros(2**(n[0]*n[1]))
    wf[index] = 1
    assert( abs( sol - energy_obs.evaluate(wf) ) < 1e-13)