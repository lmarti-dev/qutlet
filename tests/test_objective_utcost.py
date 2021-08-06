import pytest
import numpy as np
import scipy
import scipy.linalg
import cirq


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
    assert objective.evaluate(res) < 1e-10
    
@pytest.mark.parametrize(
    "t",
    [
        (0.1), (1), (-0.01)
    ],
)
def test_simulate_op(t):
    j_v = np.ones((0, 2))
    j_h = np.ones((1, 1))
    h = np.ones((1, 2))
    order = 100
    ising = Ising("GridQubit", [1, 2], j_v, j_h, h, "X", t)
    ising.set_simulator("qsim")
    ising.set_circuit("hea", {
        "parametrisation": "joint", #"layerwise",
        "p": order,
        "variables": {"x", "theta"},
        "2QubitGate": lambda theta, phi: cirq.ZZPowGate(exponent = theta, global_shift = phi)
    })
    
    objective = UtCost(ising, t, "Exact")
    params = -(2/np.pi)*t*(np.ones(2*order)/order)
    pdict = {}
    for k in range(order):
        pdict['x' + str(k)] = params[k]
        pdict['theta' + str(k)] = params[order + k]
    op = objective.simulate(
        param_resolver=pdict
    )
    assert (objective.evaluate(op) < 1e-3)

@pytest.mark.parametrize(
    "t, avg",
    [
        (0.1, True), (0.1, False), (-0.01, True)
    ],
)
def test_evaluate_batch(t, avg):
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)), "X", t)
    ising.set_Ut()
    
    objective = UtCost(ising, t, "Exact",
                       batch_wavefunctions =  Optional[np.ndarray] = None,
                       batch_averaging= avg,
                       sample_size=5)
    
    res = scipy.linalg.expm(-1j*t*hamiltonian())
    assert objective.evaluate(res) < 1e-10

def test_simulate_batch():
    return
    #assert False