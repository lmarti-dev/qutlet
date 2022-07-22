# external imports
import pytest
import numpy as np
import cirq

# internal imports
from fauvqe import CooledAdiabatic, Adiabatic, Ising, Heisenberg, HeisenbergFC, ExpectationValue
from fauvqe.utilities.generic import ptrace

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
    "field, epsilon",
    [
        (
            'X', 0
        ),
        (
            'Z', 0
        ),
        (
            'X', 1e-1
        ),
        (
            'Z', 1e-1
        )
    ]
)
def test_set_uts_w_little_cooling(field, epsilon):
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
    j_int = epsilon * np.ones((1, *n))
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
    print(out_vec)
    print(adout_vec)
    result = ptrace( out_vec.reshape(N, 1) @ out_vec.conjugate().reshape(1, N), [2, 3])
    H1.diagonalise(solver="numpy")
    #print(H1.eig_vec.transpose()[0])
    #print(adout_vec)
    print("Purity: {}".format(np.trace(result @ result)))
    if(epsilon == 0):
        assert 1 - abs(( adout_vec ).transpose().conjugate() @ result @ ( adout_vec ) ) < 1e-3
    assert 1 - abs((H1.eig_vec.transpose()[0]).transpose().conjugate() @ result @ (H1.eig_vec.transpose()[0]) ) < 1e-1

@pytest.mark.parametrize(
    "field, nbr_resets, t_steps, calc_O",
    [
        (
            'X', None,None, True
        ),
        (
            'X', 10,None, True
        ),
        (
            'X', 10,20, True
        ),
        (
            'X', None,20, True
        ),
        (
            'X', None,None, False
        ),
        (
            'X', 10,None, False
        ),
        (
            'Z', None,None, True
        )
    ]
)
def test_perform_sweep(field, nbr_resets, t_steps, calc_O):
    qubittype= "GridQubit"
    n=[2, 1]
    j_v=np.ones((1, 1))
    j_h=np.ones((2, 0))
    h= np.ones((2, 1))
    T=10
    epsilon = 1e-2
    
    zeros_v = np.zeros((n[0]-1, n[1]))
    zeros_h = np.zeros((n[0], n[1]-1))
    zeros = np.zeros((n[0], n[1]))
    
    H0 = Ising(qubittype, n, j_v, j_h, h, field)
    if(field == 'X'):
        H1 = Heisenberg(qubittype, n, j_v, j_h, zeros_v, zeros_h, zeros_v, zeros_h, zeros, h, zeros)
    else:
        H1 = Heisenberg(qubittype, n, zeros_v, zeros_h, zeros_v, zeros_h, j_v, j_h, zeros, zeros, h)
    
    m_anc = Ising("GridQubit", [1,1], np.zeros((1,1)), np.zeros((1,1)), np.ones((1,1)), 'Z')
    j_int = epsilon * np.ones((1, *n))
    int_gates = [lambda q1, q2: cirq.X(q1)*cirq.X(q2)]
    model = CooledAdiabatic(H0, H1, m_anc, int_gates, j_int, T=T)
    
    if t_steps is not None:
        model.set_Uts(t_steps)
    res, fids, energies = model.perform_sweep(nbr_resets, t_steps=t_steps, calc_O=calc_O)
    print(fids)
    print(energies)
    result = ptrace(res, [2])
    
    print("Purity: {}".format(np.trace(result @ result)))
    if model.m_sys.output is None:
        model.m_sys._set_output_state_for_sweep()
    assert 1 - abs(model.m_sys.output.transpose().conjugate() @ result @ model.m_sys.output ) < 1e-1


def test_theory_bounds():
    qubittype= "GridQubit"
    n=[2, 1]
    j_v=np.ones((1, 1))
    j_h=np.ones((2, 0))
    h= np.ones((2, 1))
    T=10
    epsilon = 1e-2
    
    zeros_v = np.zeros((n[0]-1, n[1]))
    zeros_h = np.zeros((n[0], n[1]-1))
    zeros = np.zeros((n[0], n[1]))
    
    H0 = Ising(qubittype, n, j_v, j_h, h, 'X')
    H1 = Heisenberg(qubittype, n, j_v, j_h, zeros_v, zeros_h, zeros_v, zeros_h, zeros, h, zeros)
    
    m_anc = Ising("GridQubit", [1,1], np.zeros((1,1)), np.zeros((1,1)), 
        0.5*0.475767210333732*np.ones((1,1)), 'Z')
    j_int = epsilon * np.ones((1, *n))
    int_gates = [lambda q1, q2: cirq.X(q1)*cirq.X(q2)]
    model = CooledAdiabatic(H0, H1, m_anc, int_gates, j_int, T=T)
    model.m_sys._get_minimal_energy_gap()
    model.m_anc.diagonalise()
    
    bound_dict = model.get_theory_bounds()
    sol = {
            'alpha_benchmark': 5,#__n * np.sqrt(epsilon*(1-epsilon))/(4*np.pi*epsilon) \
                #/ self.m_sys.T / self.m_sys.min_gap / S,
            'alpha_high': 2*model.m_sys.min_gap,
            'gap_difference_benchmark': 0,
            'gap_difference_high': 2*model.m_sys.min_gap,
            'dt_between_resets_benchmark': 2*np.pi/model.m_sys.min_gap,
            'dt_between_resets_high': 10
            }
    
    print(bound_dict)
    for key in bound_dict:
        if not (key == 'alpha_benchmark'):
            assert abs(bound_dict[key] - sol[key]) < 1e-7