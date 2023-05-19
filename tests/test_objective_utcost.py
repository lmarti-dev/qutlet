import pytest
import joblib
from multiprocessing import cpu_count
import numpy as np
import scipy
import scipy.linalg
import sympy
import cirq

from fauvqe import UtCost, Ising, DrivenModel, haar, haar_1qubit, uniform

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
    objective = UtCost(ising, t, 0)
    
    res = scipy.linalg.expm(-1j*t*hamiltonian())
    assert objective.evaluate(res) < 1e-10
    
@pytest.mark.parametrize(
    "t, times",
    [
        (0.1, [1]), (1, [1]), (-0.01, [1]), (0.1, [1, 3, 5]), (1, [1, 3, 5]), (-0.01, [1, 3, 5])
    ],
)
def test_simulate_op(t, times):
    j_v = np.ones((0, 2))
    j_h = np.ones((1, 1))
    h = np.ones((1, 2))
    order = 100
    ising = Ising("GridQubit", [1, 2], j_v, j_h, h, "X", t)
    ising.set_simulator("qsim")
    ising.set_circuit("hea", {
        "parametrisation": "joint", #"layerwise",
        "p": order,
        "SingleQubitVariables": [["x"]],
        "TwoQubitVariables": [["theta"]],
        "SingleQubitGates": [lambda x: cirq.PhasedXZGate(x_exponent=x, z_exponent=0, axis_phase_exponent=0)],
        "TwoQubitGates": [lambda theta: cirq.ZZPowGate(exponent = theta, global_shift = 0)]
    })
    
    objective = UtCost(ising, t, 0, time_steps=times)
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
    
    objective = UtCost(ising, t, 0, initial_wavefunctions = initials)
    
    eval_indices = np.random.randint(low=0, high=bsize, size=avg_size)
    print(eval_indices)
    res = scipy.linalg.expm(-1j*t*hamiltonian())
    outputs = np.zeros(shape=(avg_size, 4), dtype = np.complex128)
    for k in range(len(eval_indices)):
        outputs[k] = res @ initials[eval_indices[k]]
    print(outputs.shape)
    print(np.array([outputs]).shape)
    assert objective.evaluate(np.array([outputs]), options={'state_indices': eval_indices}) < 1e-10

@pytest.mark.parametrize(
    "t, m, q, times",
    [
        (0.1, 0, 1, [1]), 
        (1, 0, 1, [1, 3, 5]), 
        (-0.01, 0, 1, [1, 3, 5]), 
        (0.1, 100, 1, [1, 3, 5]), 
        (1, 100, 1, [1]), 
        (-0.01, 100, 1, [1, 3, 5]),
        (0.1, 100, 2, [1, 3, 5]), 
        (1, 10, 4, [1]), 
        (-0.01, 10, 6, [1, 3, 5])
    ],
)
@pytest.mark.higheffort
def test_simulate_batch(t, m, q, times):
    j_v = np.ones((0, 2))
    j_h = np.ones((1, 1))
    h = np.ones((1, 2))
    t_new = t - 0.01
    ex=100
    ising = Ising("GridQubit", [1, 2], j_v, j_h, h, "X", t_new)
    ising.set_simulator("qsim")
    ising.set_circuit("hea", {
        "parametrisation": "joint", #"layerwise",
        "p": ex,
        "SingleQubitVariables": [["x"]],
        "TwoQubitVariables": [["theta"]],
        "TwoQubitGates": [lambda theta: cirq.ZZPowGate(exponent = theta, global_shift = 0)],
        "SingleQubitGates": [lambda x: cirq.PhasedXZGate(x_exponent=x, z_exponent=0, axis_phase_exponent=0)],
    })
    
    bsize = 1
    initial_rands= (np.random.rand(bsize, 4)).astype(np.complex128)
    initials = np.zeros(initial_rands.shape, dtype=np.complex64)
    for k in range(bsize):
        initials[k, :] = initial_rands[k, :] / np.linalg.norm(initial_rands[k, :])
    
    params = -(2/np.pi)*t*(np.ones(2*ex)/ex)
    objective = UtCost(ising, t, m, q, initial_wavefunctions = initials, time_steps=times, use_progress_bar=True, dtype=np.complex64)

    op = objective.simulate(
        param_resolver=ising.get_param_resolver(params),
        initial_state = initials[0]
    )
    assert (objective.evaluate(np.array(op), options={'state_indices': [0]}) < 1e-3)

@pytest.mark.parametrize(
    "t, m",
    [
        (0.1, 25)
    ],
)
def test_json(t, m):
    j_v = np.ones((0, 2))
    j_h = np.ones((1, 1))
    h = np.ones((1, 2))
    ising = Ising("GridQubit", [1, 2], j_v, j_h, h, "X", t)
    bsize=1
    initial_rands= (np.random.rand(bsize, 4)).astype(np.complex128)
    initials = np.zeros(initial_rands.shape, dtype=np.complex64)
    for k in range(bsize):
        initials[k, :] = initial_rands[k, :] / np.linalg.norm(initial_rands[k, :])
    objective = UtCost(ising, t, m, initial_wavefunctions=initials)
    
    json = objective.to_json_dict()
    
    objective2 = UtCost.from_json_dict(json)
    
    assert (objective == objective2)

@pytest.mark.parametrize(
    "model, n_states, get_states_metod, m_trotter, q_trotter, t_final",
    [
        (
            Ising(  "GridQubit", 
                    [1,2], 
                    1 * np.ones((0, 2)), 
                    1 * np.ones((1, 1)), 
                    1 * np.ones((1, 2)),
                    "X"),
            8,
            uniform,
            1,
            1,
            np.pi/3
        ),
        (
            Ising(  "GridQubit", 
                    [1,2], 
                    1 * np.ones((0, 2)), 
                    1 * np.ones((1, 1)), 
                    1 * np.ones((1, 2)),
                    "X"),
            4,
            haar,
            2,
            1,
            2.747
        ),
        (
            Ising(  "GridQubit", 
                    [1,3], 
                    1 * np.ones((0, 3)), 
                    1 * np.ones((1, 2)), 
                    1 * np.ones((1, 3)),
                    "X"),
            4,
            haar_1qubit,
            2,
            2,
            2.747
        ),
        
        (
            Ising(  "GridQubit", 
                    [2,4], 
                    0.24 * np.ones((1, 4)), 
                    1 * np.ones((2, 3)), 
                    1.6 * np.ones((2, 4)),
                    "X"),
            4,
            haar,
            2,
            4,
            2.747
        ),
        (
            Ising(  "GridQubit", 
                    [3,2], 
                    1 * np.ones((2, 2)), 
                    1 * np.ones((3, 1)), 
                    1 * np.ones((3, 2)),
                    "X"),
            2,
            uniform,
            3,
            2,
            np.pi/3
        ),
        (
            Ising(  "GridQubit", 
                    [3,3], 
                    1 * np.ones((2, 3)), 
                    np.pi * np.ones((3, 2)), 
                    1 * np.ones((3, 3)),
                    "X"),
            4,
            haar_1qubit,
            2,
            2,
            2.747
        ),
        (
            DrivenModel([Ising(  "GridQubit", 
                                    [1,3], 
                                    1 * np.ones((0, 3)), 
                                    1 * np.ones((1, 2)), 
                                    0 * np.ones((1, 3)),
                                    "X"),
                        Ising(  "GridQubit", 
                                    [1,3], 
                                    0 * np.ones((0, 3)), 
                                    0 * np.ones((1, 2)), 
                                    1 * np.ones((1, 3)),
                                    "X")],
                        [lambda t : 1, lambda t : sympy.sin((2*sympy.pi/0.2)*t)],
                        T = 0.2,
                        tf = 2.747),
            4,
            haar_1qubit,
            2,
            2,
            2.747
        ),
    ],
)
@pytest.mark.higheffort
def test_vec_cost(  model, 
                    n_states, 
                    get_states_metod,
                    m_trotter, 
                    q_trotter,
                    t_final):
    model.set_simulator("cirq", {"dtype": np.complex128 })
    
    cost_states=get_states_metod(model.n[0]*model.n[1],
                                n_states)
    print(n_states)
    print(np.size(cost_states[:,0]))

    ut_cost = UtCost(   model,
                        t = t_final,
                        m = m_trotter,
                        q= q_trotter)

    ut_cost_vec = UtCost(   model,
                        t = t_final,
                        m = m_trotter,
                        q= q_trotter,
                        initial_wavefunctions=cost_states)

    model.set_circuit("trotter",
                        {"trotter_number" : m_trotter,
                        "trotter_order" : q_trotter,
                        "tf": t_final})

    cost_unitary_final_states = np.empty(shape=( cost_states.shape), dtype=np.complex128)
    for k in range(n_states): cost_unitary_final_states[k] = ut_cost._Ut @ cost_states[k]

    cost_utcost_vec_final_states = np.squeeze(ut_cost_vec._output_wavefunctions)
    
    tmp = joblib.Parallel( n_jobs=min(cpu_count(), n_states))(
                    joblib.delayed(model.simulator.simulate)
                            ( 
                                model.circuit, 
                                initial_state=cost_states[k]
                            ) for k in range(n_states)
                    )
    cost_model_final_states = np.empty(shape=( cost_states.shape), dtype=np.complex128)
    for k in range(n_states): cost_model_final_states[k] = tmp[k].state_vector() / np.linalg.norm(tmp[k].state_vector())

    cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase( cost_unitary_final_states, 
                                                                    cost_utcost_vec_final_states,
                                                                    rtol=1e-7, atol=1e-7)
                                                                
    cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase( cost_unitary_final_states, 
                                                                    cost_model_final_states,
                                                                    rtol=1e-7, atol=1e-7)

    cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase( cost_utcost_vec_final_states, 
                                                                    cost_model_final_states,
                                                                    rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize(
    "model, n_states, get_states_metod, m_trotter, q_trotter, t_final, rtol, atol",
    [
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    1 * np.ones((1, 2)), 
                    1 * np.ones((2, 1)), 
                    1 * np.ones((2, 2)),
                    "X"),
            2,
            uniform,
            250,
            2,
            np.pi/3,
            1e-5,
            1e-4
        ),
        (
            Ising(  "GridQubit", 
                    [1,3], 
                    1 * np.ones((0, 3)), 
                    1 * np.ones((1, 2)), 
                    0.1 * np.ones((1, 3)),
                    "X"),
            2,
            haar,
            150,
            2,
            np.pi/3,
            1e-6,
            1e-5
        ),
    ],
)
@pytest.mark.ultrahigheffort
def test_high_order_vec_cost(   model, 
                                n_states, 
                                get_states_metod, 
                                m_trotter, 
                                q_trotter, 
                                t_final,
                                rtol,
                                atol):
    model.set_simulator("cirq", {"dtype": np.complex128 })
    
    cost_states=get_states_metod(model.n[0]*model.n[1],
                                n_states)

    test_states = np.eye(2**(model.n[0]*model.n[1]), dtype=np.complex128)
    #test_states = cost_states

    ut_cost = UtCost(   model,
                        t = t_final,
                        m = 0,
                        q= 0,
                        initial_wavefunctions=cost_states)
    cost_unitary_final_states = np.empty(shape=( cost_states.shape), dtype=np.complex128)
    for k in range(n_states): cost_unitary_final_states[k] = ut_cost._Ut @ cost_states[k]

    ut_cost_vec = UtCost(   model,
                        t = t_final,
                        m = m_trotter,
                        q= q_trotter,
                        initial_wavefunctions=cost_states)
    cost_utcost_vec_final_states = np.squeeze(ut_cost_vec._output_wavefunctions)

    cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase( cost_unitary_final_states, 
                                                                    cost_utcost_vec_final_states,
                                                                    rtol=rtol, atol=atol)

    assert abs(ut_cost.evaluate(test_states) - ut_cost_vec.evaluate(test_states)) < rtol

@pytest.mark.parametrize(
    "model, t_final",
    [
        (
            Ising(  "GridQubit", 
                    [1,2], 
                    1 * np.ones((0, 2)), 
                    1 * np.ones((1, 1)), 
                    1 * np.ones((1, 2)),
                    "X"),
            np.pi/3,
        ),
        (
            Ising(  "GridQubit", 
                    [1,2], 
                    1 * np.ones((0, 2)), 
                    1 * np.ones((1, 1)), 
                    0 * np.ones((1, 2)),
                    "X"),
            np.pi/3,
        ),
        (
            Ising(  "GridQubit", 
                    [1,2], 
                    0 * np.ones((0, 2)), 
                    0 * np.ones((1, 1)), 
                    1 * np.ones((1, 2)),
                    "X"),
            np.pi/3,
        ),
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    1 * np.ones((1, 2)), 
                    1 * np.ones((2, 1)), 
                    1 * np.ones((2, 2)),
                    "X"),
            np.pi/3,
        ),
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    1 * np.ones((1, 2)), 
                    1 * np.ones((2, 1)), 
                    0 * np.ones((2, 2)),
                    "X"),
            np.pi/3,
        ),
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    0 * np.ones((1, 2)), 
                    0 * np.ones((2, 1)), 
                    1 * np.ones((2, 2)),
                    "X"),
            np.pi/3,
        ),
    ],
)
def test_consistency_exact_Ut(model, t_final):
    cost_states = np.eye(2**(model.n[0]*model.n[1]), dtype=np.complex128)
    test_states = np.eye(2**(model.n[0]*model.n[1]), dtype=np.complex128)

    ut_cost = UtCost(   model,
                        t = t_final,
                        m = 0,
                        q= 0)

    ut_cost_batch = UtCost(   model,
                        t = t_final,
                        m = 0,
                        q= 0,
                        initial_wavefunctions=cost_states)
    #print(np.shape(cost_states), np.shape(ut_cost_batch._output_wavefunctions), ut_cost_batch.batch_size)

    assert ut_cost.evaluate(test_states) == ut_cost_batch.evaluate(test_states, {"state_indices": range(2**(model.n[0]*model.n[1]))})

@pytest.mark.parametrize(
    "model, get_states_metod, t_final",
    [
        (
            Ising(  "GridQubit", 
                    [1,2], 
                    1 * np.ones((0, 2)), 
                    1 * np.ones((1, 1)), 
                    1 * np.ones((1, 2)),
                    "X"),
            uniform,
            np.pi/3,
        ),
        (
            Ising(  "GridQubit", 
                    [1,2], 
                    1 * np.ones((0, 2)), 
                    1 * np.ones((1, 1)), 
                    0 * np.ones((1, 2)),
                    "X"),
            uniform,
            np.pi/3,
        ),
        (
            Ising(  "GridQubit", 
                    [1,2], 
                    0 * np.ones((0, 2)), 
                    0 * np.ones((1, 1)), 
                    1 * np.ones((1, 2)),
                    "X"),
            uniform,
            np.pi/3,
        ),
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    1 * np.ones((1, 2)), 
                    1 * np.ones((2, 1)), 
                    1 * np.ones((2, 2)),
                    "X"),
            haar,
            np.pi/3,
        ),
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    1 * np.ones((1, 2)), 
                    1 * np.ones((2, 1)), 
                    0 * np.ones((2, 2)),
                    "X"),
            haar,
            np.pi/3,
        ),
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    0 * np.ones((1, 2)), 
                    0 * np.ones((2, 1)), 
                    1 * np.ones((2, 2)),
                    "X"),
            haar,
            np.pi/3,
        ),
    ],
)
def test_consistency_exact_Ut2(model, get_states_metod, t_final):
    model.set_simulator("cirq", {"dtype": np.complex128 })

    test_states=get_states_metod(model.n[0]*model.n[1],
                                2**(model.n[0]*model.n[1]))

    cost_states = np.eye(2**(model.n[0]*model.n[1]), dtype=np.complex128)

    ut_cost = UtCost(   model,
                        t = t_final,
                        m = 0,
                        q= 0)

    ut_cost_batch = UtCost(   model,
                        t = t_final,
                        m = 0,
                        q= 0,
                        initial_wavefunctions=cost_states)
    print(np.shape(cost_states), np.shape(test_states), np.shape(ut_cost_batch._output_wavefunctions), ut_cost_batch.batch_size)

    assert abs(ut_cost.evaluate(test_states) - ut_cost_batch.evaluate(test_states, {"state_indices": range(2**(model.n[0]*model.n[1]))})) < 1e-14

#This seems to consistently fail:
# Not clear why
@pytest.mark.parametrize(
    "model, get_states_method, t_final, n_states, tol",
    [
        (
            Ising(  "GridQubit", 
                    [1,2], 
                    1 * np.ones((0, 2)), 
                    1 * np.ones((1, 1)), 
                    1 * np.ones((1, 2)),
                    "X"),
            haar,
            np.pi/3,
            1024,
            1e-5
        ),
        (
            Ising(  "GridQubit", 
                    [1,2], 
                    1 * np.ones((0, 2)), 
                    1 * np.ones((1, 1)), 
                    0 * np.ones((1, 2)),
                    "X"),
            uniform,
            np.pi/3,
            16,
            1e-5
        ),
        (
            Ising(  "GridQubit", 
                    [1,2], 
                    0 * np.ones((0, 2)), 
                    0 * np.ones((1, 1)), 
                    1 * np.ones((1, 2)),
                    "X"),
            uniform,
            np.pi/3,
            16,
            1e-5
        ),
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    1 * np.ones((1, 2)), 
                    1 * np.ones((2, 1)), 
                    1 * np.ones((2, 2)),
                    "X"),
            haar,
            np.pi/3,
            16,
            1e-5
        ),
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    1 * np.ones((1, 2)), 
                    1 * np.ones((2, 1)), 
                    0 * np.ones((2, 2)),
                    "X"),
            haar,
            np.pi/3,
            16,
            1e-5
        ),
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    0 * np.ones((1, 2)), 
                    0 * np.ones((2, 1)), 
                    1 * np.ones((2, 2)),
                    "X"),
            haar,
            np.pi/3,
            16,
            1e-5
        ),
    ],
)
def test_consistency_exact_Ut3(model, get_states_method, t_final,n_states, tol):
    model.set_simulator("cirq", {"dtype": np.complex128 })

    test_states=get_states_method(model.n[0]*model.n[1],
                                2**(model.n[0]*model.n[1]))
    print(np.shape(test_states))
    print(np.linalg.norm(test_states))
    cost_states =get_states_method(model.n[0]*model.n[1],
                                n_states)
    print(np.linalg.norm(cost_states[0,:]))
    #cost_states = test_states

    ut_cost = UtCost(   model,
                        t = t_final,
                        m = 0,
                        q= 0)

    ut_cost_batch = UtCost(   model,
                        t = t_final,
                        m = 0,
                        q= 0,
                        initial_wavefunctions=cost_states)
    print(np.shape(cost_states), np.shape(test_states), np.shape(ut_cost_batch._output_wavefunctions), ut_cost_batch.batch_size)

    assert abs(ut_cost.evaluate(test_states) - ut_cost_batch.evaluate(test_states, {"state_indices": range(2**(model.n[0]*model.n[1]))})) < tol

@pytest.mark.parametrize(
    "model, t_final, m_trotter, q_trotter, tol",
    [
        (
            Ising(  "GridQubit", 
                    [1,2], 
                    1 * np.ones((0, 2)), 
                    1 * np.ones((1, 1)), 
                    1 * np.ones((1, 2)),
                    "X"),
            np.pi/3,
            35,
            2,
            1e-7,
        ),
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    1 * np.ones((2, 2)), 
                    1 * np.ones((2, 1)), 
                    0 * np.ones((2, 2)),
                    "X"),
            np.pi/3,
            1,
            1,
            1e-15,
        ),
        (
            Ising(  "GridQubit", 
                    [3,2], 
                    0 * np.ones((2, 2)), 
                    0 * np.ones((3, 1)), 
                    1 * np.ones((3, 2)),
                    "X"),
            13/7*np.pi,
            1,
            1,
            1e-15,
        ),
        (
            Ising(  "GridQubit", 
                                    [3,2], 
                                    1 * np.ones((2, 2)), 
                                    1 * np.ones((3, 1)), 
                                    1 * np.ones((3, 2)),
                                    "X"),
            np.pi/6,
            25,
            2,
            1e-7,
        ),
        #(
        #    DrivenModel([Ising(  "GridQubit", 
        #                            [3,2], 
        #                            1 * np.ones((2, 2)), 
        #                            1 * np.ones((3, 1)), 
        #                            0 * np.ones((3, 2)),
        #                            "X"),
        #                Ising(  "GridQubit", 
        #                            [3,2], 
        #                            0 * np.ones((2, 2)), 
        #                            0 * np.ones((3, 1)), 
        #                            1 * np.ones((3, 2)),
        #                            "X")],
        #                [lambda t : 1, lambda t : 1],
        #                T=0.2),
        #    np.pi/6,
        #    25,
        #    2,
        #    1e-7,
        #),
    ],
)
def test_consistency_high_order_trotter(model, t_final, m_trotter, q_trotter, tol):
    model.t = t_final
    del model.circuit
    model.circuit= cirq.Circuit()
    #print( type(model) )
    #print(model.__dict__)
    print( model.hamiltonian(t_final) )
    print("model.circuit: {}:".format(model.circuit))
    if isinstance(model, DrivenModel):
        model.set_Ut(m=1, q=1)
    else:
        model.set_Ut()
    print(np.shape(model._Ut))
    unitary_exact_cost = UtCost(   model,
                        t = t_final,
                        m = 0,
                        q= 0)

    unitary_Trotter_cost = UtCost(   model,
                        t = t_final,
                        m = m_trotter,
                        q= q_trotter)

    assert unitary_exact_cost.evaluate(unitary_Trotter_cost._Ut) < tol
    del model
    del unitary_exact_cost
    del unitary_Trotter_cost

    #This is the same
    #assert unitary_Trotter_cost.evaluate(unitary_exact_cost._Ut) < tol
#############################################################
#                                                           #
#                     Test errors                           #
#                                                           #
#############################################################
class MockUtCost(UtCost):
    def __init__(self):
        return
    
def test_abstract_gradient_optimiser():
    with pytest.raises(NotImplementedError):
        MockUtCost().evaluate()

def test_no_odd_order():
    t=0.1
    m=10
    q=3
    times=[1]
    j_v = np.ones((0, 2))
    j_h = np.ones((1, 1))
    h = np.ones((1, 2))
    t_new = t - 0.01
    ex=100
    ising = Ising("GridQubit", [1, 2], j_v, j_h, h, "X", t_new)
    
    initial_rands= (np.random.rand(1, 4)).astype(np.complex128)
    
    with pytest.raises(NotImplementedError):
        objective = UtCost(ising, t, m, q, initial_wavefunctions = initial_rands, time_steps=times, use_progress_bar=True, dtype=np.complex64)

"""
    Old test of simulating np.complex128 wavefunctions
@pytest.mark.parametrize(
    "n",
    [
        (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17)
    ],
)
def test_128(n):
    n = int(n)
    t=0.1
    order = 100
    j_v = np.ones((0, n))
    j_h = np.ones((1, n))
    h = np.ones((1, n))
    ising = Ising("GridQubit", [1, n], j_v, j_h, h, "X", t)
    bsize=1
    initial_rands= (np.random.rand(bsize, 2**n)).astype(np.complex128)
    initials = np.zeros(initial_rands.shape, dtype=np.complex64)
    for k in range(bsize):
        initials[k, :] = initial_rands[k, :] / np.linalg.norm(initial_rands[k, :])
    objective = UtCost(ising, t, order, batch_wavefunctions=initials)
    assert False
"""