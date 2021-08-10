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
    "t, avg_size",
    [
        (0.1, 1), (0.1, 5), (-0.01, 1)
    ],
)
def test_evaluate_batch(t, avg_size):
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)), "X", t)
    ising.set_Ut()
    
    bsize = 10
    initial_rands= (np.random.rand(bsize, 4)).astype(np.complex128)
    initials = np.zeros(initial_rands.shape, dtype=np.complex128)
    for k in range(bsize):
        initials[k, :] = initial_rands[k, :] / np.linalg.norm(initial_rands[k, :])
    
    objective = UtCost(ising, t, "Exact", batch_wavefunctions = initials)
    
    eval_indices = np.random.randint(low=0, high=bsize, size=avg_size)
    print(eval_indices)
    res = scipy.linalg.expm(-1j*t*hamiltonian())
    outputs = np.zeros(shape=(avg_size, 4), dtype = np.complex128)
    for k in range(len(eval_indices)):
        outputs[k] = res @ initials[eval_indices[k]]
    assert objective.evaluate(outputs, eval_indices) < 1e-10

@pytest.mark.parametrize(
    "t,U",
    [
        (0.1, "Exact"), (1, "Exact"), (-0.01, "Exact"), (0.1, "Trotter"), (1, "Trotter"), (-0.01, "Trotter")
    ],
)
def test_simulate_batch(t, U):
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
    
    bsize = 1
    initial_rands= (np.random.rand(bsize, 4)).astype(np.complex128)
    initials = np.zeros(initial_rands.shape, dtype=np.complex64)
    for k in range(bsize):
        initials[k, :] = initial_rands[k, :] / np.linalg.norm(initial_rands[k, :])
    
    params = -(2/np.pi)*t*(np.ones(2*order)/order)
    objective = UtCost(ising, t, U, ising, params, batch_wavefunctions = initials)
    op = objective.simulate(
        param_resolver=ising.get_param_resolver(params),
        initial_state = initials[0]
    )
    assert (objective.evaluate([op], [0]) < 1e-3)

