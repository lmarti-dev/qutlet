from operator import mod
import cirq
import pytest
import numpy as np
from timeit import default_timer
from typing import List

from fauvqe import AbstractExpectationValue, ExpectationValue, Ising, IsingXY, Heisenberg

def consistency_check(model):
    #Get random state
    wf = np.random.rand(2**(model.n[0]*model.n[1]))
    wf = wf.astype(np.complex64)/np.linalg.norm(wf)
    wf = wf/np.linalg.norm(wf)
    #Create ising object
    
    EV_obj =ExpectationValue(model)
    AEV_obj =AbstractExpectationValue(model)

    print("n: {}\tExpectationValue: {}\tAbstractExpectationValue/n: {}\trel. difference: {}"\
    .format(model.n, EV_obj.evaluate(wf), AEV_obj.evaluate(wf, atol=1e-16)/(model.n[0]*model.n[1]), \
        abs(EV_obj.evaluate(wf)-AEV_obj.evaluate(wf,atol=1e-16)/(model.n[0]*model.n[1]))/abs(EV_obj.evaluate(wf))))
    assert abs(EV_obj.evaluate(wf)-AEV_obj.evaluate(wf,atol=1e-16)/(model.n[0]*model.n[1])) < 1e-6


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
def test_consistency_Ising(n, b, field):
    #Create ising object
    model = Ising(  "GridQubit", 
                    n, 
                    2*(np.random.rand(n[0]-b[0],n[1])-0.5),
                    2*(np.random.rand(n[0],n[1]-b[1])-0.5), 
                    2*(np.random.rand(*n)-0.5),
                    field)

    consistency_check(model)

@pytest.mark.parametrize(
    "n, b, field",
    [
        ([3,3], [1,1], "X"),
        ([3,3], [1,1], "Z"),
    ],
)
def test_consistency_IsingXY(n, b, field):
    #Create IsingXY object
    model = IsingXY(  "GridQubit", 
                    n, 
                    2*(np.random.rand(n[0]-b[0],n[1])-0.5),
                    2*(np.random.rand(n[0],n[1]-b[1])-0.5), 
                    2*(np.random.rand(n[0]-b[0],n[1])-0.5),
                    2*(np.random.rand(n[0],n[1]-b[1])-0.5), 
                    2*(np.random.rand(*n)-0.5),
                    field)
    
    consistency_check(model)

@pytest.mark.parametrize(
    "n, b",
    [
        ([3,3], [1,1]),
    ],
)
def test_consistency_Heisenberg(n, b):
    #Create Heisenberg object
    model = Heisenberg(  "GridQubit", 
                    n, 
                    2*(np.random.rand(n[0]-b[0],n[1])-0.5),
                    2*(np.random.rand(n[0],n[1]-b[1])-0.5), 
                    2*(np.random.rand(n[0]-b[0],n[1])-0.5),
                    2*(np.random.rand(n[0],n[1]-b[1])-0.5), 
                    2*(np.random.rand(n[0]-b[0],n[1])-0.5),
                    2*(np.random.rand(n[0],n[1]-b[1])-0.5), 
                    2*(np.random.rand(*n)-0.5),
                    2*(np.random.rand(*n)-0.5),
                    2*(np.random.rand(*n)-0.5),
                )
    
    consistency_check(model)

@pytest.mark.parametrize(
    "state, expectation_value",
    [
        (np.array([1, 0, 0, 0], dtype=np.complex64), -0.5),
        (1/np.sqrt(2) * np.array([0, 1, 1, 0], dtype=np.complex64), 0.5)
    ],
)
def test_evaluate(state, expectation_value):
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = ExpectationValue(ising)
    
    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values)
    )
    expval = objective.evaluate(state)
    assert abs(expval - expectation_value) < 1e-7

@pytest.mark.parametrize(
    "n, qubit_order",
    [
        (
            [1,2],
            None
        ),
        (
            [1,2],
            {cirq.GridQubit(0, 0): 0, cirq.GridQubit(0, 1): 1}
        ),
        #This should fail, but does not?
        #(
        #    [2,2],
        #    {cirq.GridQubit(0, 0): 0, cirq.GridQubit(0, 1): 3, cirq.GridQubit(1, 0): 2, cirq.GridQubit(1, 1): 1}
        #),
    ],
)
def test_evaluate_rot_qubit_order(n, qubit_order):
    j_v0 = 2*(np.random.rand(n[0]-1,n[1])- 0.5)
    j_h0 = 2*(np.random.rand(n[0],n[1]-1)- 0.5)
    h0 = 2*(np.random.rand(n[0],n[1])- 0.5)

    ising = Ising("GridQubit", n, j_v0, j_h0, h0, "X")
    ising.set_circuit("basics",{    "start": "identity"})
    ising.set_simulator("cirq")
    
    wavefunction = np.random.rand(2**(n[0]*n[1]))
    wavefunction = wavefunction/np.linalg.norm(wavefunction)

    objective = ExpectationValue(ising)
    print(objective._observable)
    assert abs(objective.evaluate(wavefunction) - 
                objective.evaluate(wavefunction, {"rotation_circuits": ising.circuit,
                                                  "qubit_order": qubit_order  })) < 1e-7

@pytest.mark.parametrize(
    "n",
    [
        (
            [1,2]
        ),
        #(
        #    [2,1]
        #),
        #(
        #    [1,3]
        #),
        #(
        #    [3,1]
        #),
        #(
        #    [2,2]
        #),
        #(
        #    [3,2]
        #),
    ],
)
def test_evaluate_rot_X(n):
    j_v0 = 2*(np.random.rand(n[0]-1,n[1])- 0.5)
    j_h0 = 2*(np.random.rand(n[0],n[1]-1)- 0.5)
    #j_v0 = np.zeros((n[0]-1,n[1]))
    #j_h0 = np.zeros((n[0],n[1]-1))
    h0 = 2*(np.random.rand(n[0],n[1])- 0.5)

    ising = Ising("GridQubit", n, j_v0, j_h0, h0, "X")
    ising.set_circuit("basics",{    "start": "identity"})
    rot_circuits=[ising.circuit.copy()]
    rot_circuits.append(ising.circuit.copy())
    ising.set_simulator("cirq")
    
    wavefunction = np.random.rand(2**(n[0]*n[1]))
    wavefunction = wavefunction/np.linalg.norm(wavefunction)

    objective = ExpectationValue(ising)
    print(objective._observable)
    assert abs(objective.evaluate(wavefunction) - objective.evaluate(wavefunction, {"rotation_circuits": rot_circuits})) < 1e-7

@pytest.mark.parametrize(
    "n",
    [
        (
            [1,2]
        ),
        (
            [2,1]
        ),
        (
            [1,3]
        ),
        (
            [3,1]
        ),
        (
            [2,2]
        ),
        (
            [3,2]
        ),
    ],
)
def test_evaluate_rot_Z(n):
    j_v0 = 2*(np.random.rand(n[0]-1,n[1])- 0.5)
    j_h0 = 2*(np.random.rand(n[0],n[1]-1)- 0.5)
    h0 = 2*(np.random.rand(n[0],n[1])- 0.5)

    ising = Ising("GridQubit", n, j_v0, j_h0, h0, "Z")
    ising.set_circuit("basics",{    "start": "identity"})
    ising.set_simulator("cirq")
    
    wavefunction = np.random.rand(2**(n[0]*n[1]))
    wavefunction = wavefunction/np.linalg.norm(wavefunction)

    objective = ExpectationValue(ising)
    assert abs(objective.evaluate(wavefunction) - objective.evaluate(wavefunction, {"rotation_circuits": ising.circuit})) < 1e-7

def test_json():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = ExpectationValue(ising)
    print(objective)
    json = objective.to_json_dict()
    
    objective2 = ExpectationValue.from_json_dict(json)
    
    assert (objective == objective2)


def test_simulate():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = ExpectationValue(ising)

    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values)
    )

#################################################################
#                                                               #
#                       Assert tests                            #
#                                                               #
#################################################################
def test_notimplemented_evaluate():
    n = [2, 2]
    b = [1, 1]
    model = Ising(  "GridQubit", 
                    n, 
                    2*(np.random.rand(n[0]-b[0],n[1])-0.5),
                    2*(np.random.rand(n[0],n[1]-b[1])-0.5), 
                    2*(np.random.rand(*n)-0.5))
    model.energy_fields = ["blub", "blub"]
    obj = ExpectationValue(model)
    with pytest.raises(NotImplementedError):
        obj.evaluate(np.random.rand(16))

def test_notimplemented_rotate():
    n = [2, 2]
    b = [1, 1]
    model = Ising(  "GridQubit", 
                    n, 
                    2*(np.random.rand(n[0]-b[0],n[1])-0.5),
                    2*(np.random.rand(n[0],n[1]-b[1])-0.5), 
                    2*(np.random.rand(*n)-0.5))
    obj = ExpectationValue(model)
    with pytest.raises(NotImplementedError):
        bases = ["blub" for k in range(4)]
        bases[0] = "Z"
        obj._rotate(np.random.rand(16), bases)

def test_assert_evaluate_rotate():
    n = [2, 2]
    b = [1, 1]
    model = Ising(  "GridQubit", 
                    n, 
                    2*(np.random.rand(n[0]-b[0],n[1])-0.5),
                    2*(np.random.rand(n[0],n[1]-b[1])-0.5), 
                    2*(np.random.rand(*n)-0.5))
    obj = ExpectationValue(model)

    wavefunction = np.random.rand(2**(n[0]*n[1]))
    wavefunction = wavefunction/np.linalg.norm(wavefunction)
    
    with pytest.raises(AssertionError):
        obj.evaluate(wavefunction, {"rotation_circuits": -1})

    model.set_simulator("cirq")
    with pytest.raises(AssertionError):
        obj.evaluate(wavefunction, {"rotation_circuits": [0,-1]})

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