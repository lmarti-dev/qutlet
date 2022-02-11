# external imports
import pytest
import numpy as np
import cirq
from scipy.linalg import expm
import sympy

# internal imports
from fauvqe import HeisenbergFC, ExpectationValue

def test__eq__():
    n = [1,3];
    model = HeisenbergFC("GridQubit", n, np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])))
    model.set_circuit("qaoa")
    
    model2 = HeisenbergFC("GridQubit", n, np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])))
    model2.set_circuit("qaoa")

    assert (model == model2)

    model.set_Ut()
    assert model != model2 


@pytest.mark.parametrize(
    "qubittype, n, j_x, j_y, j_z, h_x, h_y, h_z",
    [
        #############################################################
        #                   2 qubit tests                           #
        #############################################################
        (
            "GridQubit",
            [1, 2],
            np.ones((1, 2, 1, 2)) / 5,
            np.ones((1, 2, 1, 2)) / 5,
            np.ones((1, 2, 1, 2)) / 5,
            np.zeros((1, 2)) / 10,
            np.zeros((1, 2)) / 10,
            np.zeros((1, 2)) / 10,
        ),
        (
            "GridQubit",
            [2, 1],
            np.ones((2, 1, 2, 1)) / 2,
            np.ones((2, 1, 2, 1)) / 2,
            np.ones((2, 1, 2, 1)) / 2,
            np.zeros((2, 1)) / 10,
            np.zeros((2, 1)) / 10,
            np.zeros((2, 1)) / 10,
        ),
        (
            "GridQubit",
            [1, 2],
            np.zeros((1, 2, 1, 2)),
            np.zeros((1, 2, 1, 2)),
            np.zeros((1, 2, 1, 2)),
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
            np.zeros((2, 2, 2, 2)) / 2,
            np.zeros((2, 2, 2, 2)) / 2,
            np.zeros((2, 2, 2, 2)) / 2,
            np.ones((2, 2)) / 3,
            np.ones((2, 2)) / 3,
            np.ones((2, 2)) / 3,
        ),
        (
            "GridQubit",
            [2, 2],
            np.zeros((2, 2, 2, 2)) / 2,
            np.ones((2, 2, 2, 2)) / 2,
            np.ones((2, 2, 2, 2)) / 5,
            np.ones((2, 2)) / 3,
            np.ones((2, 2)) / 3,
            np.ones((2, 2)) / 3,
        ),
    ],
)
def test_copy(qubittype, n, j_x, j_y, j_z, h_x, h_y, h_z):
    model = HeisenbergFC(qubittype, n, j_x, j_y, j_z, h_x, h_y, h_z)
    model.set_circuit("qaoa")
    model2 = model.copy()
    
    #Test whether the objects are the same
    assert( model == model2 )
    
    #But there ID is different
    assert( model is not model2 )

def test_json():
    model = HeisenbergFC("GridQubit", [1, 2], np.zeros((1, 2, 1, 2)), np.ones((1, 2, 1, 2)), np.zeros((1, 2, 1, 2)), np.ones((1, 2)), np.zeros((1, 2)), np.ones((1, 2)))
    
    json = model.to_json_dict()
    
    model2 = HeisenbergFC.from_json_dict(json)
    
    assert (model == model2)


@pytest.mark.parametrize(
    "qubittype, n, j_x, j_y, j_z, h_x, h_y, h_z, sol",
    [
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2, 2, 2)),
            np.ones((2, 2, 2, 2)),
            np.ones((2, 2, 2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            -1.5
        ),
        (
            "GridQubit",
            [2, 2],
            np.zeros((2, 2, 2, 2)),
            np.zeros((2, 2, 2, 2)),
            np.zeros((2, 2, 2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            -1
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2, 2, 2)),
            np.ones((2, 2, 2, 2)),
            np.ones((2, 2, 2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            -2.5
        )]
)
def test_energy(qubittype, n, j_x, j_y, j_z, h_x, h_y, h_z, sol):
    model = HeisenbergFC(qubittype, n, j_x, j_y, j_z, h_x, h_y, h_z)
    obj = ExpectationValue(model)
    ini = np.zeros(16).astype(np.complex64)
    ini[0] = 1
    assert abs(obj.evaluate(ini) - sol) < 1e-13