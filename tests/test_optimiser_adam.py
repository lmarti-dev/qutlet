"""
    Use QAOA Ising to test if ADAM optimiser works
    1.create Ising object + simple 4 qubit QAOA?
    2. set_optimiser
    3.ising.optimise()

    Later: possibly put into test class 

    26.03.2020:
        test_optimse currently fails as energy() is jZZ-hX energy in optimiser
        but one needs to give in this case the optimiser energy(.., field='Z')
        Needs to be fixed by generalisation of Ising.set_simulator()

    08.04.21: Need to add some results/run time test
"""
# external imports
import pytest
import numpy as np
import cirq

from time import time, localtime, strftime
from multiprocessing import cpu_count

# internal imports
from fauvqe import Ising, ADAM, ExpectationValue, UtCost


def test_set_optimiser():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_circuit("qaoa", {"p": 1})
    adam = ADAM()
    objective = ExpectationValue(ising)
    adam.optimise(objective, n_jobs=1)


# This is potentially a higher effort test:
#############################################################
#                                                           #
#                  Sequential version                       #
#                                                           #
#############################################################
@pytest.mark.higheffort
def test_optimise():
    ising = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
        "Z"
    )
    ising.set_circuit("qaoa", {"p": 2, "H_layer": False})
    ising.set_circuit_param_values(0.3 * np.ones(np.size(ising.circuit_param)))
    eps = 10 ** -3
    exp_val_z = ExpectationValue(ising)
    adam = ADAM({
        'eps':eps,
        'break_param':25,
        'eta': 4 * 10 ** -2,
    }
    )
    res = adam.optimise(exp_val_z, n_jobs=1)

    #wf = ising.simulator.simulate(
    #    ising.circuit,
    #    param_resolver=ising.get_param_resolver(ising.circuit_param_values),
    #).state_vector()
    # Result smaller than -0.5 up to eta
    res.get_latest_step().reset_objective()
    assert -0.5 > res.get_latest_objective_value() - eps
    # Result smaller than -0.5 up to eta


@pytest.mark.higheffort
def test_adam_multiple_models_and_auto_joblib():
    ising1 = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
    )
    ising1.set_circuit("qaoa", {
        "p": 2
    })
    ising1.set_circuit_param_values(0.3 * np.ones(np.size(ising1.circuit_param)))
    ising2 = Ising(
        "GridQubit",
        [1, 2],
        np.ones((0, 2)),
        np.ones((1, 1)),
        np.ones((1, 2)),
    )
    ising2.set_circuit("qaoa", {"p": 1})

    adam = ADAM()

    objective1 = ExpectationValue(ising1)

    res1 = adam.optimise(objective1, n_jobs=-1)

    objective2 = ExpectationValue(ising2)
    res2 = adam.optimise(objective2, n_jobs=-1)

    print(res1, res2)

#############################################################
#                                                           #
#                    Joblib version                         #
#                                                           #
#############################################################
@pytest.mark.higheffort
def test_optimise_joblib():
    ising = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
        "Z"
    )
    ising.set_circuit("qaoa", {"p": 2, "H_layer": False})
    ising.set_circuit_param_values(0.3 * np.ones(np.size(ising.circuit_param)))
    adam = ADAM({
        'break_param':25,
        'eta': 4e-2,
    }
    )
    expval_z = ExpectationValue(ising)

    res = adam.optimise(expval_z, n_jobs=-1)
    wavefunction = expval_z.simulate(
        param_resolver=ising.get_param_resolver(res.get_latest_step().params)
    )

    # Result smaller than -0.5 up to eta
    assert -0.5 > expval_z.evaluate(wavefunction) - adam.options['eps']
    # Result smaller than -0.5 up to eta

@pytest.mark.parametrize(
    "sym",
    [
        (True),(False)
    ],
)
def test_optimise_no_simulator_change(sym):
    ising = Ising(
        "GridQubit", [2, 2], 0.1 * np.ones((1, 2)), 0.5 * np.ones((2, 1)), 0.2 * np.ones((2, 2))
    )
    ising.set_circuit("qaoa", {"p": 2})
    ising.set_circuit_param_values(0.3 * np.ones(np.size(ising.circuit_param)))
    ising.set_simulator(simulator_name = "cirq")
    
    adam = ADAM({
        'break_param':1,
        'eta': 4e-2,
        'symmetric_gradient': sym
    })
    expval_z = ExpectationValue(ising)
    #assert(ising.simulator == 0)

    res = adam.optimise(expval_z, n_jobs=-1)
    assert(ising.simulator == adam._objective.model.simulator)

@pytest.mark.higheffort
def test__get_single_energy():
    ising = Ising(
        "GridQubit", [2, 2], 0.1 * np.ones((1, 2)), 0.5 * np.ones((2, 1)), 0.2 * np.ones((2, 2))
    )
    ising.set_circuit("qaoa", {"p": 2})
    ising.set_circuit_param_values(0.3 * np.ones(np.size(ising.circuit_param)))

    adam = ADAM({
        'break_param': 1,
        'eta': 4e-2
    })
    expval_z = ExpectationValue(ising)
    res = adam.optimise(expval_z, n_jobs=-1)
    gg_gradients, cost = adam._get_gradients(adam._objective.model.circuit_param_values, 8)

    # 2 layer, 2 parameters, 2 energies each
    single_energies = np.zeros(2*2*2)
    for j in range(8):
        single_energies[j] = adam._get_single_cost(
            {**{str(adam._circuit_param[i]): adam._objective.model.circuit_param_values[i] for i in range(adam._n_param)}} , 
            j)
    single_energies = np.array(single_energies).reshape((adam._n_param, 2)) 
    se_gradients = np.matmul(single_energies, np.array((1, -1))) / (2 * adam.options['eps']) 
    np.testing.assert_allclose(gg_gradients    , se_gradients, rtol=1e-15, atol=1e-15)

@pytest.mark.parametrize(
    "sym, n_jobs, sim",
    [
        (True, -1, 'qsim'),(False, -1, 'qsim')
    ],
)
@pytest.mark.higheffort
def test_optimise_batch(sym, n_jobs, sim):
    t=0.5
    ising = Ising("GridQubit", [1, 4], np.ones((0, 4)), np.ones((1, 4)), np.ones((1,4)), "X", t)
    ising.set_Ut()
    ising.set_circuit("hea", {
        "parametrisation": "joint", #"layerwise",
        "p": 3,
        "2Qvariables": [["theta"]],
        "2QubitGates": [lambda theta: cirq.ZZPowGate(exponent = theta, global_shift = 0)],
        "1QubitGates": None
    })
    ising.set_circuit_param_values(-(2/np.pi)*t/3 *np.ones(np.size(ising.circuit_param)))
    ising.set_simulator(sim)
    bsize = 100
    initial_rands= (np.random.rand(bsize, 16)).astype(np.complex128)
    initials = np.zeros(initial_rands.shape, dtype=np.complex64)
    for k in range(bsize):
        initials[k, :] = initial_rands[k, :] / np.linalg.norm(initial_rands[k, :])
    
    objective = UtCost(ising, t, 0, initial_wavefunctions = initials)
    
    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values), initial_state=initials[0]
    )
    trotter_cost = ( objective.evaluate(np.array([wavefunction]), options={'indices': [0]}) )
    #print(trotter_cost)
    adam = ADAM({
        'break_param': 100,
        'batch_size': 1, 
        'eps': 1e-5, 
        'eta': 1e-2,
        'symmetric_gradient': sym, 
        'use_progress_bar': True
    })
    #print(objective.model.circuit_param_values.view())
    res = adam.optimise(objective, n_jobs=n_jobs)
    #print(res.get_latest_step().params)
    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(res.get_latest_step().params), initial_state=initials[0]
    )
    var_cost = (objective.evaluate(np.array([wavefunction]), options={'indices': [0]}))
    #print(var_cost)
    assert var_cost/10 < trotter_cost

@pytest.mark.skipif(cpu_count() < 4, reason="No speed-up expected for single core machine")
@pytest.mark.higheffort
def test_times():
    start = time()
    test_optimise()
    end = time()
    
    time_seq = end - start
    
    start = time()
    test_optimise_joblib()
    end = time()
    
    time_par = end - start
    
    textfile = open("./tests/performance/times.txt", "a")
    textfile.write('==='+ str(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())) +'===\n')
    textfile.write('Test No Indices \n\n')
    textfile.write('Sequential Time: ' + str(time_seq) +'\n')
    textfile.write('Parallel Time: ' + str(time_par) +'\n\n')
    textfile.close()
    
    assert 2*time_par < time_seq, 'No speedup through parallelisation'

@pytest.mark.skipif(cpu_count() < 4, reason="No speed-up expected for single core machine")
@pytest.mark.parametrize(
    "sym, sim",
    [
        (True, 'qsim'),
        (False, 'qsim'),
        (True, 'cirq'),
        (False, 'cirq')
    ],
)
@pytest.mark.higheffort
def test_times_batch(sym, sim):
    start = time()
    test_optimise_batch(sym, 1, sim)
    end = time()
    
    time_seq = end - start
    
    start = time()
    test_optimise_batch(sym, -1, sim)
    end = time()
    
    time_par = end - start
    
    textfile = open("./tests/performance/times.txt", "a")
    textfile.write('==='+ str(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())) +'===\n')
    textfile.write('Test Indices \n\n')
    textfile.write('Sequential Time: ' + str(time_seq) +'\n')
    textfile.write('Parallel Time: ' + str(time_par) +'\n\n')
    textfile.close()
    
    assert time_par < time_seq, 'No speedup through parallelisation'

@pytest.mark.skipif(cpu_count() < 4, reason="No speed-up expected for single core machine")
@pytest.mark.parametrize(
    "sym, sim",
    [
        (True, 'qsim'),
        (False, 'qsim'),
        (True, 'cirq'),
        (False, 'cirq')
    ],
)
@pytest.mark.higheffort
def test_times_adam_ExpVal(sym, sim):
    ising = Ising("GridQubit", [1, 4], np.ones((0, 4)), np.ones((1, 4)), np.ones((1,4)), "X")
    ising.set_circuit("hea", {"p": 3})
    ising.set_circuit_param_values(-(2/np.pi)/3 *np.ones(np.size(ising.circuit_param)))
    print("np.size(ising.circuit_param): {}".format(np.size(ising.circuit_param)))
    ising.set_simulator(sim)
    objective = ExpectationValue(ising)
    adam = ADAM({
        'break_param': 100,
        'eps': 1e-5, 
        'eta': 1e-2,
        'symmetric_gradient': sym, 
        'use_progress_bar': False
    })
    

    start = time()
    res = adam.optimise(objective, n_jobs=1)
    end = time()
    time_seq = end - start
    
    #reset
    ising.set_circuit_param_values(-(2/np.pi)/3 *np.ones(np.size(ising.circuit_param)))
    objective = ExpectationValue(ising)
    start = time()
    res = adam.optimise(objective, n_jobs=-1)
    end = time()
    
    time_par = end - start
    
    textfile = open("./tests/performance/times.txt", "a")
    textfile.write('==='+ str(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())) +'===\n')
    textfile.write('Test Adam + ExpectationValue \n\n')
    textfile.write('Sequential Time: ' + str(time_seq) +'\n')
    textfile.write('Parallel Time: ' + str(time_par) +'\n\n')
    textfile.close()
    
    assert time_par < time_seq, 'No speedup through parallelisation'

def test_json():
    t=0.1
    j_v = np.ones((0, 2))
    j_h = np.ones((1, 1))
    h = np.ones((1, 2))
    ising = Ising("GridQubit", [1, 2], j_v, j_h, h, "X", t)
    bsize=10
    initial_rands= (np.random.rand(bsize, 4)).astype(np.complex128)
    initials = np.zeros(initial_rands.shape, dtype=np.complex64)
    for k in range(bsize):
        initials[k, :] = initial_rands[k, :] / np.linalg.norm(initial_rands[k, :])
    objective = UtCost(ising, t, 0, initial_wavefunctions=initials)
    adam = ADAM()
    json = adam.to_json_dict()
    
    adam2 = ADAM.from_json_dict(json)
    
    assert adam.__eq__(adam2)

#############################################################
#                                                           #
#                     Test errors                           #
#                                                           #
#############################################################
def test_adam_break_cond_assert():
    with pytest.raises(AssertionError):
        ADAM({'break_cond':"atol"})
