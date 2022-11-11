import pytest
import cirq
import numpy as np
import sympy

from fauvqe import Ising, ObjectiveSum, ExpectationValue, Correlation, Variance, flatten

def _get_euclidian_vec(n,i):
    tmp = np.zeros(2**n)
    tmp[i] = 1
    return tmp

@pytest.mark.parametrize(
    "model, x0_value, ground_truth",
    [
        (
            Ising(  "GridQubit", 
                    [2, 3], 
                    np.ones((1, 3)), 
                    np.ones((2, 2)), 
                    np.ones((2, 3)),
                    "X"),
            0,
            _get_euclidian_vec(6,0)
        ),
        (
            Ising(  "GridQubit", 
                    [2, 3], 
                    np.ones((1, 3)), 
                    np.ones((2, 2)), 
                    np.ones((2, 3)),
                    "X"),
            1,
            _get_euclidian_vec(6,-1)
        ),  
    ]
)
def test_simulate(model, x0_value, ground_truth):
    print(type(model))
    model.set_circuit("hea",
                        {   "SingleQubitVariables": ['a', 'x', 'z' ],
                            "SingleQubitGates": [lambda a, x, z: cirq.XPowGate(exponent=x)],
                            "TwoQubitGates" : []})
    objective0 = Correlation(model)
    objective1 = Variance(  model, 
                            observables=objective0._observable)

    objectivesum_obj = ObjectiveSum([objective0, objective1])
    wavefunctions = objectivesum_obj.simulate(param_resolver={"x0": x0_value})
    for wavefunction in wavefunctions:
        print(ground_truth)
        print(wavefunction)
        assert 1- abs(np.vdot(wavefunction, ground_truth)) < 1e-14

""" 
@pytest.mark.parametrize(
    "n",
    [
        (
            2
        ), 
        (
            [2, 2]
        ),  
        (
            4
        ), 
        (
            [4, 4]
        ),
        (
            8
        ), 
        (
            [8, 8]
        ), 
        (
            16
        ), 
        (
            [16, 16]
        ), 
    ],
)
def test_evaluate(n):
    #use here that for Pauli strings var = 1 - <O>²
    np.set_printoptions(precision=16)
    #Generate Ising Object as MockAbstractModel
    mockmodel = Ising("GridQubit", [1, 1], np.ones((0, 1)), np.ones((1, 1)), np.ones((1, 1)))

    #Generate random state vector or unitary
    if isinstance(n,int):
        rand_matrix = 2*(np.random.rand(n)-0.5) + 2j*(np.random.rand(n)-0.5)
        rand_matrix /= np.linalg.norm(rand_matrix)
    else:
        rand_matrix = unitary_group.rvs(n[0])

    #Generate MatrixCost object and test by evaluating it by comparing 
    #rand_matrix with itself
    objective = MatrixCost(mockmodel, rand_matrix)
    assert objective.evaluate(rand_matrix) < 1e-15

    #Generate 2nd random state vector or unitary
    #Check wether MatrixCost \in [0,1]
    if isinstance(n,int):
        rand_matrix2 = 2*(np.random.rand(n)-0.5) + 2j*(np.random.rand(n)-0.5)
        rand_matrix2 /= np.linalg.norm(rand_matrix2)
    else:
        rand_matrix2 = unitary_group.rvs(n[0])

    assert 0 <= objective.evaluate(rand_matrix2) <= 1
    
@pytest.mark.parametrize(
    "model, circuit, param_resolver, solution",
    [
        (
            [
                Ising("GridQubit", [1, 2], np.zeros((0, 2)), np.zeros((1, 1)), np.ones((1, 2))),
                cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0))**2, cirq.H.on(cirq.GridQubit(0, 1))**2),
                cirq.ParamResolver(),
                np.array((1, 0, 0,0)),
            ]
        ),
        (
            [
                Ising("GridQubit", [1, 2], np.zeros((0, 2)), np.zeros((1, 1)), np.ones((1, 2))),
                cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0))**2, cirq.H.on(cirq.GridQubit(0, 1))**2),
                cirq.ParamResolver(),
                np.identity(4)
            ]
        ),
        (
            [
                Ising("GridQubit", [1, 2], np.zeros((0, 2)), np.zeros((1, 1)), np.ones((1, 2))),
                cirq.Circuit(cirq.X.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1))**2),
                cirq.ParamResolver(),
                np.array((0, 0, 1,0)),
            ]
        ),
        (
            [
                Ising("GridQubit", [1, 2], np.zeros((0, 2)), np.zeros((1, 1)), np.ones((1, 2))),
                cirq.Circuit(cirq.X.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1))**2),
                cirq.ParamResolver(),
                np.array(  ((0, 0, 1,0),
                            (0, 0, 0,1),
                            (1, 0, 0,0),
                            (0, 1, 0,0)))
            ]
        ),
        (
            [
                Ising("GridQubit", [1, 2], np.zeros((0, 2)), np.zeros((1, 1)), np.ones((1, 2))),
                cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.CNOT.on(cirq.GridQubit(0, 0),cirq.GridQubit(0, 1))),
                cirq.ParamResolver(),
                np.array((np.sqrt(2)/2, 0, 0,np.sqrt(2)/2), dtype=np.float128),
            ]
        ),
        (
            [
                Ising("GridQubit", [1, 2], np.zeros((0, 2)), np.zeros((1, 1)), np.ones((1, 2))),
                cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.CNOT.on(cirq.GridQubit(0, 0),cirq.GridQubit(0, 1))),
                cirq.ParamResolver(),
                np.array(  ((np.sqrt(2)/2   , 0             , np.sqrt(2)/2  ,0              ),
                            (0              , np.sqrt(2)/2  , 0             ,np.sqrt(2)/2   ),
                            (0              , np.sqrt(2)/2  , 0             ,-np.sqrt(2)/2   ),
                            (np.sqrt(2)/2   , 0             , -np.sqrt(2)/2  ,0              )))
            ]
        ),
    ],
)
def test_simulate(model, circuit, param_resolver, solution):
    print(circuit)
    print(model.qubits)
    print(solution)
    model.circuit = circuit
    objective = MatrixCost(model, solution)

    #Generate MatrixCost object and use its simulate() function
    wf = objective.simulate(param_resolver)
    print(np.round(wf, decimals=3))

    #assert whether given solution and circuit-generated unitary/state_vector are equal up to tolerance
    assert objective.evaluate(wf) < 1e-15

@pytest.mark.parametrize(
    "n",
    [
        (
            2
        ), 
        (
            [2, 2]
        ),  
        (
            4
        ), 
        (
            [4, 4]
        ),
    ],
)
def test__repr__(n):
    mockmodel = Ising("GridQubit", [1, 1], np.ones((0, 1)), np.ones((1, 1)), np.ones((1, 1)))

    #Generate random state vector or unitary
    if isinstance(n,int):
        rand_matrix = np.random.rand(n) + 1j*np.random.rand(n)
        rand_matrix /= np.linalg.norm(rand_matrix)
    else:
        rand_matrix = unitary_group.rvs(n[0])

    #Generate MatrixCost object and test by evaluating it by comparing 
    #rand_matrix with itself
    objective = MatrixCost(mockmodel, rand_matrix)
    assert repr(objective) == "<MatrixCost matrix={}>".format(rand_matrix)

@pytest.mark.parametrize(
    "n",
    [
        (
            2
        ), 
        (
            [2, 2]
        ),  
        (
            4
        ), 
        (
            [4, 4]
        ),
    ],
)
def test_json(n):
    mockmodel = Ising("GridQubit", [1, 1], np.ones((0, 1)), np.ones((1, 1)), np.ones((1, 1)))

    #Generate random state vector or unitary
    if isinstance(n,int):
        rand_matrix = np.random.rand(n) + 1j*np.random.rand(n)
        rand_matrix /= np.linalg.norm(rand_matrix)
    else:
        rand_matrix = unitary_group.rvs(n[0])

    #Generate MatrixCost object and test by comparing it
    #with itself after to_json_dict and from_json_dict
    objective = MatrixCost(mockmodel, rand_matrix)
    json = objective.to_json_dict()
    objective2 = MatrixCost.from_json_dict(json)
    assert (objective == objective2)

#############################################################
#                                                           #
#                    Assert tests                           #
#                                                           #
#############################################################
def test_asserts():
    mockmodel = Ising("GridQubit", [1, 1], np.ones((0, 1)), np.ones((1, 1)), np.ones((1, 1)))

    #__init__ asserts
    with pytest.raises(AssertionError):
        objective = MatrixCost(mockmodel, np.random.rand(3))

    with pytest.raises(AssertionError):
        objective = MatrixCost(mockmodel, np.identity(3))

    with pytest.raises(AssertionError):
        objective = MatrixCost(mockmodel, np.random.rand(2,2,2))

"""