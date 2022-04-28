# external imports
import pytest
import numpy as np
import cirq
from scipy.linalg import expm
import sympy

# internal imports
from fauvqe import Adiabatic, Ising, Heisenberg, HeisenbergFC, ExpectationValue

def test__eq__():
    n = [1,3]; boundaries = [1, 0]
    H0 = Heisenberg("GridQubit", 
                       n, 
                       np.ones((n[0]-boundaries[0], n[1])), 
                    np.ones((n[0], n[1]-boundaries[1])), 
                       np.ones((n[0]-boundaries[0], n[1])), 
                    np.ones((n[0], n[1]-boundaries[1])), 
                       np.ones((n[0]-boundaries[0], n[1])), 
                    np.ones((n[0], n[1]-boundaries[1])), 
                       np.ones((n[0], n[1])),
                       np.ones((n[0], n[1])),
                       np.ones((n[0], n[1])))
    
    H1 = Ising("GridQubit", 
               n, 
               np.ones((n[0]-boundaries[0], n[1])), 
               np.ones((n[0], n[1]-boundaries[1])), 
               np.ones((n[0], n[1])),
               "X"
              )
    
    model = Adiabatic(H0, H1)
    model2 = Adiabatic(H0, H1)
    
    assert (model == model2)
    
    model.set_Ut()
    assert model != model2 


@pytest.mark.parametrize(
    "qubittype, n, j_x_v, j_x_h, j_y_v, j_y_h, j_z_v, j_z_h, h_x, h_y, h_z",
    [
        #############################################################
        #                   2 qubit tests                           #
        #############################################################
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.zeros((1, 2)) / 10,
            np.zeros((1, 2)) / 10,
            np.zeros((1, 2)) / 10,
        ),
        (
            "GridQubit",
            [2, 1],
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            np.zeros((2, 1)) / 10,
            np.zeros((2, 1)) / 10,
        ),
        (
            "GridQubit",
            [1, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((1, 2)) / 5,
            np.zeros((1, 2)) / 2,
            np.zeros((1, 2)) / 5,
            np.zeros((1, 2)) / 2,
            np.zeros((1, 2)) / 5,
            np.ones((1, 2)) / 3,
            np.ones((1, 2)) / 3,
            np.ones((1, 2)) / 3,
        ),
        #############################################################
        #                   4 qubit tests                           #
        #############################################################
        (
            "GridQubit",
            [2, 2],
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.ones((2, 2)) / 3,
            np.ones((2, 2)) / 3,
            np.ones((2, 2)) / 3,
        ),
        (
            "GridQubit",
            [2, 2],
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.ones((2, 2)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((2, 2)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((2, 2)) / 3,
            np.ones((2, 2)) / 3,
            np.ones((2, 2)) / 3,
        ),
    ],
)
def test_copy(qubittype, n, j_x_v, j_x_h, j_y_v, j_y_h, j_z_v, j_z_h, h_x, h_y, h_z):
    H0 = Heisenberg(qubittype, n, j_x_v, j_x_h, j_y_v, j_y_h, j_z_v, j_z_h, h_x, h_y, h_z)
    H1 = Ising(qubittype, n, j_z_v, j_z_h, h_x, "X")
    
    model = Adiabatic(H0, H1)
    model.set_circuit("qaoa")
    model2 = model.copy()
    
    #Test whether the objects are the same
    assert( model == model2 )
    
    #But there ID is different
    assert( model is not model2 )

def test_json():
    H0 = Heisenberg("GridQubit", [1, 2], np.zeros((0, 2)), np.ones((1, 1)), np.zeros((0, 2)), np.ones((1, 1)), np.zeros((0, 2)), np.ones((1, 1)), np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2)))
    
    H1 = Heisenberg("GridQubit", [1, 2], np.zeros((0, 2)), np.ones((1, 1)), np.zeros((0, 2)), np.ones((1, 1)), np.zeros((0, 2)), np.ones((1, 1)), np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2)))
    
    model = Adiabatic(H0, H1)
    
    json = model.to_json_dict()
    
    model2 = Adiabatic.from_json_dict(json)
    
    assert (model == model2)

@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, t, field, sol",
    [
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)),
            np.ones((1, 1)),
            np.ones((1, 2)),
            0,
            'X',
            -0.5
        ),
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)),
            np.ones((1, 1)),
            np.ones((1, 2)),
            1,
            'X',
            0.0
        ),
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)),
            np.ones((1, 1)),
            np.ones((1, 2)),
            0.5,
            'X',
            -0.25
        ),
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)),
            np.ones((1, 1)),
            np.ones((1, 2)),
            0.5,
            'Z',
            -1
        )
    ]
)
def test_energy(qubittype, n, j_v, j_h, h, t, field, sol):
    zeros_v = np.zeros((n[0]-1, n[1]))
    zeros_h = np.zeros((n[0], n[1]-1))
    zeros = np.zeros((n[0], n[1]))
    H0 = Ising(qubittype, n, j_v, j_h, h, field)
    if(field == 'X'):
        H1 = Heisenberg(qubittype, n, j_v, j_h, zeros_v, zeros_h, zeros_v, zeros_h, zeros, zeros, zeros)
    else:
        H1 = Heisenberg(qubittype, n, zeros_v, zeros_h, zeros_v, zeros_h, j_v, j_h, zeros, zeros, zeros)
    
    model = Adiabatic(H0, H1, t=t)
    obj = ExpectationValue(model)
    ini = np.zeros(2**(n[0]*n[1])).astype(np.complex64)
    ini[0] = 1
    assert abs(obj.evaluate(ini) - sol) < 1e-13