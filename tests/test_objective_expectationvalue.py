import pytest
import numpy as np
from timeit import default_timer

from fauvqe import AbstractExpectationValue ,ExpectationValue, Ising


@pytest.mark.parametrize(
    "state, res",
    [
        (np.array([1, 0, 0, 0], dtype=np.complex64), -0.5),
        (1/np.sqrt(2) * np.array([0, 1, 1, 0], dtype=np.complex64), 0.5)
    ],
)
def test_evaluate(state, res):
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = ExpectationValue(ising)
    
    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values)
    )
    expval = objective.evaluate(state)
    assert abs(expval - res) < 1e-7

def test_simulate():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = ExpectationValue(ising)

    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values)
    )

def test_json():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = ExpectationValue(ising)
    print(objective)
    json = objective.to_json_dict()
    
    objective2 = ExpectationValue.from_json_dict(json)
    
    assert (objective == objective2)


@pytest.mark.parametrize(
    "n, b, field",
    [
        ([3,3], [1,1], "X"),
        ([3,3], [0,1], "X"),
        ([3,3], [1,0], "X"),
        ([3,3], [0,0], "X"),
        ([3,3], [1,1], "Z"),
        ([3,3], [0,1], "Z"),
        ([3,3], [1,0], "Z"),
        ([3,3], [0,0], "Z"),
    ],
)
def test_consistency(n, b, field):
    #Get random state
    wf = np.random.rand(2**(n[0]*n[1]))
    wf = wf.astype(np.complex64)/np.linalg.norm(wf)
    wf = wf/np.linalg.norm(wf)
    #Create ising object
    ising = Ising(  "GridQubit", 
                    n, 
                    2*(np.random.rand(n[0]-b[0],n[1])-0.5),
                    2*(np.random.rand(n[0],n[1]-b[1])-0.5), 
                    2*(np.random.rand(*n)-0.5),
                    field)

    EV_obj =ExpectationValue(ising)
    AEV_obj =AbstractExpectationValue(ising)

    print("n: {}\tExpectationValue: {}\tAbstractExpectationValue/n: {}\trel. difference: {}"\
    .format(n, EV_obj.evaluate(wf), AEV_obj.evaluate(wf, atol=1e-16)/(n[0]*n[1]), \
        abs(EV_obj.evaluate(wf)-AEV_obj.evaluate(wf,atol=1e-16)/(n[0]*n[1]))/abs(EV_obj.evaluate(wf))))
    assert abs(EV_obj.evaluate(wf)-AEV_obj.evaluate(wf,atol=1e-16)/(n[0]*n[1])) < 1e-6

# This works when executed in main
# but somehoe not with pytest
"""
def test_speed_up():
    n = [3,3]; b = [0,0]; field = "X"; min_speed_up = 120; reps = 100
    #Get random state
    wf = np.random.rand(2**(n[0]*n[1]))
    wf = wf.astype(np.complex64)/np.linalg.norm(wf)
    wf = wf/np.linalg.norm(wf)
    #Create ising object
    ising = Ising(  "GridQubit", 
                    n, 
                    2*(np.random.rand(n[0]-b[0],n[1])-0.5),
                    2*(np.random.rand(n[0],n[1]-b[1])-0.5), 
                    2*(np.random.rand(*n)-0.5),
                    field)

    EV_obj =ExpectationValue(ising)
    AEV_obj =AbstractExpectationValue(ising)

    t0 = default_timer()
    for i in range(reps):
        tmp = EV_obj.evaluate(wf)
    t1 = default_timer()
    for i in range(reps):
        tmp = AEV_obj.evaluate(wf)
    t2 = default_timer()
    print("ExpectationValue: {}s\tAbstractExpectationValue: {}s\nCustom speed-up: {}".format(t1-t0, t2 -t1, (t2-t1)/(t1-t0)) )

    assert (t2-t1)/(t1-t0) > min_speed_up
"""