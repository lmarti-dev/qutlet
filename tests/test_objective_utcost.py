import pytest
import numpy as np
import scipy
import scipy.linalg

from fauvqe import UtCost, Ising

def hamiltonian():
    Z = np.array([[1, 0],
                  [0, -1]])
    X = np.array([[0, 1], 
                  [1, 0]])
    return -np.kron(Z, Z) - np.kron(X, np.eye(2)) - np.kron(np.eye(2), X)
    
@pytest.mark.parametrize(
    "t",
    [
        (200*np.pi), (0.1), (15), (-0.01)
    ],
)
def test_evaluate_op(t):
    np.set_printoptions(precision=16)
    
    #Define reference system
    mat = hamiltonian()
    vals, vecs = scipy.linalg.eigh(mat)
    mat_exp = scipy.linalg.expm(-1j*t*mat)
    mat_diag = vecs @ np.diag(np.exp(-1j*t*vals)) @ np.matrix.getH(vecs)
    
    #Compare with fauvqe
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)), "X", t)
    ising.set_Ut()
    #print(ising._Ut.shape)
    #print(ising._Ut)
    #print(mat_diag.shape)
    #print(mat_diag)
    #print("np.linalg.norm(ising._Ut - mat_diag): {}".format(np.linalg.norm(ising._Ut - mat_diag)))
    objective = UtCost(ising, t, "Exact")
    
    res = scipy.linalg.expm(-1j*t*hamiltonian())
    assert objective.evaluate(res) < 1e-7
    
@pytest.mark.parametrize(
    "t",
    [
        (0.1), #(15), (-0.01)
    ],
)
def test_simulate_op(t):
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)), "X", t)
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    ising.set_Ut()
    
    objective = UtCost(ising, t, "Exact")
    
    op = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values)
    )
    print(objective.evaluate(op))
    return
    #assert False

def test_evaluate_batch_avg():
    return
    #assert False

def test_evaluate_batch_random():
    return
    #assert False

def test_simulate_batch_avg():
    return
    #assert False
    
def test_simulate_batch_random():
    return
    #assert False