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
            -np.array([[1, 1, 1, 0], 
                      [1, -1, 0, 1], 
                      [1, 0, -1, 1], 
                      [0, 1, 1, 1]])
        ),
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)),
            np.ones((1, 1)),
            np.ones((1, 2)),
            1.0,
            'X',
            -np.array([[0, 0, 0, 1], 
                      [0, 0, 1, 0], 
                      [0, 1, 0, 0], 
                      [1, 0, 0, 0]])
        ),
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)),
            np.ones((1, 1)),
            np.ones((1, 2)),
            0.5,
            'X',
            -0.5*np.array([[1, 1, 1, 1], 
                      [1, -1, 1, 1], 
                      [1, 1, -1, 1], 
                      [1, 1, 1, 1]])
        ),
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)),
            np.ones((1, 1)),
            np.ones((1, 2)),
            0.5,
            'Z',
            - 0.5 * np.array([[4, 0, 0, 0], 
                              [0, -2, 0, 0], 
                              [0, 0, -2, 0], 
                              [0, 0, 0, 0]])
        )
    ]
)
def test_set_hamiltonian_overwrite(qubittype, n, j_v, j_h, h, t, field, sol):
    zeros_v = np.zeros((n[0]-1, n[1]))
    zeros_h = np.zeros((n[0], n[1]-1))
    zeros = np.zeros((n[0], n[1]))
    H0 = Ising(qubittype, n, j_v, j_h, h, field)
    if(field == 'X'):
        H1 = Heisenberg(qubittype, n, j_v, j_h, zeros_v, zeros_h, zeros_v, zeros_h, zeros, zeros, zeros)
    else:
        H1 = Heisenberg(qubittype, n, zeros_v, zeros_h, zeros_v, zeros_h, j_v, j_h, zeros, zeros, zeros)
    
    model = Adiabatic(H0, H1, t=t)
        
    print(model._hamiltonian.matrix())
    assert np.linalg.norm(model._hamiltonian.matrix() - sol) < 1e-13


@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, T, field",
    [
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)),
            np.ones((1, 1)),
            np.ones((1, 2)),
            100,
            'X',
        ),
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)),
            np.ones((1, 1)),
            np.ones((1, 2)),
            100,
            'Z'
        )
    ]
)
def test_set_uts(qubittype, n, j_v, j_h, h, T, field):
    zeros_v = np.zeros((n[0]-1, n[1]))
    zeros_h = np.zeros((n[0], n[1]-1))
    zeros = np.zeros((n[0], n[1]))
    H0 = Ising(qubittype, n, j_v, j_h, h, field)
    if(field == 'X'):
        H1 = Heisenberg(qubittype, n, j_v, j_h, zeros_v, zeros_h, zeros_v, zeros_h, zeros, h, zeros)
    else:
        H1 = Heisenberg(qubittype, n, zeros_v, zeros_h, zeros_v, zeros_h, j_v, j_h, zeros, zeros, h)
    
    model = Adiabatic(H0, H1, T=T)
    model.set_Uts()
    N=2**(n[0]*n[1])
    res = np.eye(N)
    for i in range(len(model._Uts)):
        res = model._Uts[i] @ res
    H0.diagonalise(solver="numpy")
    initial = H0.eig_vec.transpose()[0]
    out = res @ initial
    H1.diagonalise(solver="numpy")
    print(out)
    print(H1.eig_vec.transpose()[0])
    assert 1-abs((H1.eig_vec.transpose()[0]).transpose().conjugate() @ out ) < 1e-3

@pytest.mark.parametrize(
    "field",
    [
        (
            'X'
        ),
        (
            'Z'
        )
    ]
)
def test_set_initial_and_output_state(field):
    qubittype = "GridQubit"
    n = [1, 2]
    j_v = np.ones((0, 2))
    j_h = np.ones((1, 1))
    h = np.ones((1, 2))
    T = 100
            
    zeros_v = np.zeros((n[0]-1, n[1]))
    zeros_h = np.zeros((n[0], n[1]-1))
    zeros = np.zeros((n[0], n[1]))
    H0 = Ising(qubittype, n, j_v, j_h, h, field)
    if(field == 'X'):
        H1 = Heisenberg(qubittype, n, j_v, j_h, zeros_v, zeros_h, zeros_v, zeros_h, zeros, h, zeros)
    else:
        H1 = Heisenberg(qubittype, n, zeros_v, zeros_h, zeros_v, zeros_h, j_v, j_h, zeros, zeros, h)
    
    model = Adiabatic(H0, H1, T=T)
    model._set_initial_state_for_sweep()
    H0.diagonalise(solver="numpy")
    initial = H0.eig_vec.transpose()[0]
    assert 1-abs(model.initial.conjugate().transpose() @ initial) < 1e-7
    
    model._set_output_state_for_sweep()
    H1.diagonalise(solver="numpy")
    output = H1.eig_vec.transpose()[0]
    assert 1-abs(model.output.conjugate().transpose() @ output) < 1e-7

def test_get_minimal_energy_gap():
    qubittype = "GridQubit"
    n = [1, 1]
    h = np.ones(shape=(1, 1))
    T = 100
    
    zeros_v = np.zeros((n[0]-1, n[1]))
    zeros_h = np.zeros((n[0], n[1]-1))
    
    H0 = Ising(qubittype, n, zeros_v, zeros_h, h, 'Z')
    H1 = Ising(qubittype, n, zeros_v, zeros_h, h, 'X')
    
    model = Adiabatic(H0, H1, T=T)
    
    model.get_minimal_energy_gap()
    print(model.gaps)
    assert abs(model.min_gap - np.sqrt(2)) < 1e-7
    #Test whether it does not change anything a second time
    model.get_minimal_energy_gap()
    assert abs(model.min_gap - np.sqrt(2)) < 1e-7
    
    
@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, T, field",
    [
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)),
            np.ones((1, 1)),
            np.ones((1, 2)),
            100,
            'X'
        ),
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)),
            np.ones((1, 1)),
            np.ones((1, 2)),
            100,
            'Z'
        )
    ]
)
def test_set_ut_custom_sweep(qubittype, n, j_v, j_h, h, T, field):
    zeros_v = np.zeros((n[0]-1, n[1]))
    zeros_h = np.zeros((n[0], n[1]-1))
    zeros = np.zeros((n[0], n[1]))
    H0 = Ising(qubittype, n, j_v, j_h, h, field)
    if(field == 'X'):
        H1 = Heisenberg(qubittype, n, j_v, j_h, zeros_v, zeros_h, zeros_v, zeros_h, zeros, h, zeros)
    else:
        H1 = Heisenberg(qubittype, n, zeros_v, zeros_h, zeros_v, zeros_h, j_v, j_h, zeros, zeros, zeros)
    
    for sweep in [lambda time: time**2/T**2, lambda time: 4*(time/T - 0.5)**3 + 0.5]:
        model = Adiabatic(H0, H1, sweep, T=T)
        model.set_Uts()
        N=2**(n[0]*n[1])
        res = np.eye(N)
        for i in range(len(model._Uts)):
            res = model._Uts[i] @ res
        H0.diagonalise(solver="numpy")
        initial = H0.eig_vec.transpose()[0]
        out = res @ initial
        H1.diagonalise(solver="numpy")
        print(out)
        print(H1.eig_vec.transpose()[0])
        assert 1-abs((H1.eig_vec.transpose()[0]).transpose().conjugate() @ out ) < 1e-2

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