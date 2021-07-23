import pytest
import numpy as np
import scipy

from fauvqe import UtCost, Ising

def hamiltonian():
    Z = np.array([[1, 0],
                  [0, -1]])
    X = np.array([[0, 1], 
                  [1, 0]])
    return -0*np.kron(Z, Z) - np.kron(X, np.eye(2)) - np.kron(np.eye(2), X)
    
@pytest.mark.parametrize(
    "t",
    [
        (0.1), #(15), (-0.01)
    ],
)
def test_evaluate_op(t):
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), 0*np.ones((1, 1)), np.ones((1, 2)), "X", t)
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    ising.set_Ut()
    print(ising._Ut)
    objective = UtCost(ising, t, "Exact")
    
    res = scipy.linalg.expm(-1j*t*hamiltonian())
    print(res)
    acc = objective.evaluate(res)
    print(acc)
    assert acc < 1e-10

@pytest.mark.parametrize(
    "t",
    [
        (0.1), #(15), (-0.01)
    ],
)
def test_simulate_op(t):
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), 0*np.ones((1, 1)), np.ones((1, 2)), "X", t)
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    ising.set_Ut()
    
    objective = UtCost(ising, t, "Exact")
    
    op = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values)
    )
    print(objective.evaluate(op))
    assert False

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