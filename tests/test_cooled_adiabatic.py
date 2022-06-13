# external imports
import pytest
import numpy as np
import cirq
from scipy.linalg import expm
import sympy

# internal imports
from fauvqe import CooledAdiabatic, Adiabatic, Ising, Heisenberg, HeisenbergFC, ExpectationValue
from fauvqe.utils import ptrace

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
                       np.ones((n[0], n[1]))
                   )
    
    H1 = Ising("GridQubit", 
               n, 
               np.ones((n[0]-boundaries[0], n[1])), 
               np.ones((n[0], n[1]-boundaries[1])), 
               np.ones((n[0], n[1])),
               "X"
              )
    
    m_anc = Ising("GridQubit", [1,n[1]], np.zeros((1,n[1])), np.zeros((1,n[1])), np.ones((1,n[1])))
    j_int = np.ones((1, *n))
    int_gates = [lambda q1, q2: cirq.X(q1)*cirq.X(q2)]
    
    model = CooledAdiabatic(H0, H1, m_anc, int_gates, j_int)
    model2 = CooledAdiabatic(H0, H1, m_anc, int_gates, j_int)
    
    assert (model == model2)
    
    model.set_Uts()
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
    
    m_anc = Ising("GridQubit", [1,n[1]], np.zeros((1,n[1])), np.zeros((1,n[1])), np.ones((1,n[1])))
    j_int = np.ones((1, *n))
    int_gates = [lambda q1, q2: cirq.X(q1)*cirq.X(q2)]
    
    model = CooledAdiabatic(H0, H1, m_anc, int_gates, j_int)
    model.set_circuit("qaoa")
    model2 = model.copy()
    
    #Test whether the objects are the same
    assert( model == model2 )
    
    #But there ID is different
    assert( model is not model2 )

def test_json():
    n = [1, 2]
    H0 = Heisenberg("GridQubit", [1, 2], np.zeros((0, 2)), np.ones((1, 1)), np.zeros((0, 2)), np.ones((1, 1)), np.zeros((0, 2)), np.ones((1, 1)), np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2)))
    
    H1 = Heisenberg("GridQubit", [1, 2], np.zeros((0, 2)), np.ones((1, 1)), np.zeros((0, 2)), np.ones((1, 1)), np.zeros((0, 2)), np.ones((1, 1)), np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2)))
    
    m_anc = Ising("GridQubit", [1,n[1]], np.zeros((1,n[1])), np.zeros((1,n[1])), np.ones((1,n[1])))
    j_int = np.ones((1, *n))
    int_gates = [lambda q1, q2: cirq.X(q1)*cirq.X(q2)]
    
    model = CooledAdiabatic(H0, H1, m_anc, int_gates, j_int)
    
    json = model.to_json_dict()
    
    model2 = CooledAdiabatic.from_json_dict(json)
    
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
def test_set_hamiltonian_override(qubittype, n, j_v, j_h, h, t, field, sol):
    zeros_v = np.zeros((n[0]-1, n[1]))
    zeros_h = np.zeros((n[0], n[1]-1))
    zeros = np.zeros((n[0], n[1]))
    N = 2**(n[0]*n[1])
    X = np.array([
        [0, 1],
        [1, 0]
    ])
    
    H0 = Ising(qubittype, n, j_v, j_h, h, field)
    if(field == 'X'):
        H1 = Heisenberg(qubittype, n, j_v, j_h, zeros_v, zeros_h, zeros_v, zeros_h, zeros, zeros, zeros)
    else:
        H1 = Heisenberg(qubittype, n, zeros_v, zeros_h, zeros_v, zeros_h, j_v, j_h, zeros, zeros, zeros)
    
    m_anc = Ising("GridQubit", [1,n[1]], np.zeros((1,n[1])), np.zeros((1,n[1])), np.ones((1,n[1])), 'Z')
    j_int = np.ones((1, *n))
    int_gates = [lambda q1, q2: cirq.X(q1)*cirq.X(q2)]
    model = CooledAdiabatic(H0, H1, m_anc, int_gates, j_int, t=t)
    
    solution = np.kron(sol, np.eye(N)) + np.kron(np.eye(N), m_anc.hamiltonian.matrix()) - np.kron(np.eye(2), np.kron(X, np.kron(np.eye(2), X))) - np.kron(X, np.kron(np.eye(2), np.kron(X, np.eye(2))))
    print(solution - (model.hamiltonian.matrix()))
    print(model.hamiltonian)
    assert np.linalg.norm(model.hamiltonian.matrix() - solution ) < 1e-13


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
def test_set_uts_wo_cooling(field):
    qubittype= "GridQubit"
    n=[1, 2]
    j_v=np.ones((0, 2))
    j_h=np.ones((1, 1))
    h= np.ones((1, 2))
    T=10
    
    zeros_v = np.zeros((n[0]-1, n[1]))
    zeros_h = np.zeros((n[0], n[1]-1))
    zeros = np.zeros((n[0], n[1]))
    
    H0 = Ising(qubittype, n, j_v, j_h, h, field)
    if(field == 'X'):
        H1 = Heisenberg(qubittype, n, j_v, j_h, zeros_v, zeros_h, zeros_v, zeros_h, zeros, h, zeros)
    else:
        H1 = Heisenberg(qubittype, n, zeros_v, zeros_h, zeros_v, zeros_h, j_v, j_h, zeros, zeros, h)
    
    m_anc = Ising("GridQubit", [1,n[1]], np.zeros((1,n[1])), np.zeros((1,n[1])), np.ones((1,n[1])), 'Z')
    j_int = np.zeros((1, *n))
    int_gates = [lambda q1, q2: cirq.X(q1)*cirq.X(q2)]
    model = CooledAdiabatic(H0, H1, m_anc, int_gates, j_int, T=T)
    model.set_Uts()
    admodel = Adiabatic(H0, H1, T=T)
    admodel.set_Uts()
    
    N=2**(2*n[0]*n[1])
    adN=2**(n[0]*n[1])
    res = np.eye(N)
    adres = np.eye(adN)
    for i in range(len(model._Uts)):
        res = model._Uts[i] @ res
        adres = admodel._Uts[i] @ adres
    H0.diagonalise(solver="numpy")
    initial = H0.eig_vec.transpose()[0]
    adout_vec = adres @ initial
    initial = np.kron(initial, np.array([1, 0, 0, 0]))
    out_vec = res @ initial
    result = ptrace( out_vec.reshape(N, 1) @ out_vec.conjugate().reshape(1, N), [2, 3])
    H1.diagonalise(solver="numpy")
    print(H1.eig_vec.transpose()[0])
    print(adout_vec)
    assert 1 - abs(( adout_vec ).transpose().conjugate() @ result @ ( adout_vec ) ) < 1e-3
    assert 1 - abs((H1.eig_vec.transpose()[0]).transpose().conjugate() @ result @ (H1.eig_vec.transpose()[0]) ) < 1e-3