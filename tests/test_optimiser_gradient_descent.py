"""
    Use QAOA Ising to test if gradient descent optimiser works
    1.create Ising object + simple 4 qubit QAOA?
    2. set_optimiser
    3.ising_obj.optimise()

    Later: possibly put into test class 

    26.03.2020:
        test_optimse currently fails as energy() is jZZ-hX energy in optimiser
        but one needs to give in this case the optimiser energy(.., field='Z')
        Needs to be fixed by generalisation of Ising.set_simulator()

"""
# external imports
import pytest
import numpy as np

import cirq

# internal imports
from fauvqe import Ising, GradientDescent, ExpectationValue, UtCost
from fauvqe import IsingXY, AbstractExpectationValue, SpinModel

#This test misses a real assert
@pytest.mark.higheffort
def test_set_optimiser():
    ising_obj = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    #ising_obj = SpinModel(
    #    "GridQubit", 
    #    [1, 2], 
    #    [np.ones((0, 2))], 
    #    [np.ones((1, 1))], 
    #    [np.ones((1, 2))], 
    #    [lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)], 
    #    [cirq.X]
    #)
    ising_obj.set_circuit("qaoa", {"p": 1})
    gd = GradientDescent()
    obj = ExpectationValue(ising_obj)
    #obj = AbstractExpectationValue(ising_obj)
    gd.optimise(obj)

    #Add pro forma assert:
    assert True


# This is potentially a higher effort test:
@pytest.mark.higheffort
def test_optimise():
    ising_obj = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
        "Z"
    )
    ising_obj.set_circuit("qaoa", {"p": 2, "H_layer": False})
    ising_obj.set_circuit_param_values(0.3 * np.ones(np.size(ising_obj.circuit_param)))
    eta = 2e-2
    gd = GradientDescent({
        'break_param':25,
        'eta':eta,
    })
    obj = ExpectationValue(ising_obj)
    res = gd.optimise(obj)

    final_step = res.get_latest_step()

    assert -0.5 > final_step.objective - eta
    # Result smaller than -0.5 up to eta

@pytest.mark.higheffort
def test_optimise_batch():
    t=0.5
    ising = Ising("GridQubit", [1, 4], np.ones((0, 4)), np.ones((1, 4)), np.ones((1,4)), "X", t)
    ising.set_Ut()
    ising.set_circuit("hea", {
        "parametrisation": "joint", #"layerwise",
        "p": 3,
    })
    ising.set_circuit_param_values(-(2/np.pi)*t/3 *np.ones(np.size(ising.circuit_param)))
    
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
    print(trotter_cost)
    gd = GradientDescent({
        'break_param': 100,
        'batch_size': 1,
        'eps': 1e-5,
        'eta': 1e-2,
        'symmetric_gradient': False
    })
    print(objective.model.circuit_param_values.view())
    res = gd.optimise(objective)
    print(res.get_latest_step().params)
    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(res.get_latest_step().params), initial_state=initials[0]
    )
    var_cost = (objective.evaluate(np.array([wavefunction]), options={'indices': [0]}))
    print(var_cost)
    assert var_cost/10 < trotter_cost

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
    gd = GradientDescent()
    json = gd.to_json_dict()
    
    gd2 = GradientDescent.from_json_dict(json)
    
    assert gd == gd2

#############################################################
#                     Test errors                           #
#############################################################
def test_GradientDescent_break_cond_assert():
    with pytest.raises(AssertionError):
        GradientDescent({'break_cond': "atol"})