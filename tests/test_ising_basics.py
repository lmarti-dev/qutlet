# external imports
import pytest
import numpy as np
import cirq
import sympy
import itertools

# internal imports
from fauvqe import Ising, SpinModel
from fauvqe.models.circuits.basics import SpinModelDummy
from .test_isings import IsingTester

"""
What to test:

 (x)   Class IsingDummy
 (x)   _exact_layer
 (0)   _hadamard_layer
 (0)   _mf_layer
 (0)   _neel_layer
 (0)   add_missing_cpv
 (0)  rm_unused_cpv
 (0)  set_circuit
"""
@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, field",
    [
        ("GridQubit",
            [2, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((2, 1)) / 5,
            np.ones((2, 2)) / 3,
            "X"),
        ("GridQubit",
            [2, 2],
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.ones((2, 2)) / 3,
            "Z")
    ]
)
def test_IsingDummy(qubittype, n, j_v, j_h, h, field):
    #Deduce correct function of IsingDummy to correct function of Ising:
    ising= Ising(qubittype, n, j_v, j_h, h, field)
    if(field == "X"):
        one_q_gate = [cirq.X]
    elif(field == "Z"):
        one_q_gate = [cirq.Z]
    ising_dummy= SpinModelDummy(qubittype, n, [j_v], [j_h], [h], [lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)], one_q_gate)
    #Do these asserts to recognise if Ising() is changed
    assert(ising.n == ising_dummy.n).all()
    assert(ising.j_h == ising_dummy.j_h).all()
    assert(ising.j_v == ising_dummy.j_v).all()
    assert(ising.h == ising_dummy.h).all()
    assert ising.circuit_param == ising_dummy.circuit_param
    assert(ising.circuit_param_values == ising_dummy.circuit_param_values).all()
    assert ising.circuit == ising_dummy.circuit
    assert ising.qubittype == ising_dummy.qubittype
    assert ising.simulator.__class__ == ising_dummy.simulator.__class__

    #These are the important asserts:
    assert ising.hamiltonian == ising_dummy.hamiltonian
    assert (ising.hamiltonian.matrix() == ising_dummy.hamiltonian.matrix()).all()

@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, field, n_exact, location",
    [
        (
            "GridQubit",
            [2, 2],
            np.ones((1, 2)) / 2,
            np.ones((2, 1)) / 5,
            np.ones((2, 2)) / 3,
            "X",
            [2, 2],
            "start"
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((1, 2)) / 2,
            np.zeros((2, 1)) / 5,
            np.ones((2, 2)) / 3,
            "X",
            [2, 1],
            "start"
            ),
        (
            "GridQubit",
            [3, 2],
            np.ones((3, 2)) / 2,
            np.zeros((3, 1)) / 5,
            np.ones((3, 2)) / 3,
            "Z",
            [3, 1],
            "end"
        ),
    ]
)
def test__exact_layer(qubittype, n, j_v, j_h, h, field, n_exact, location):
    ising= Ising(qubittype, n, j_v, j_h, h, field)
    ising.set_circuit("basics",{    location: "exact", 
                                    "n_exact": n_exact})
    #print(ising.circuit)
    wf = ising.simulator.simulate(ising.circuit).state_vector()
    wf = wf/np.linalg.norm(wf)
    
    #Note that the test states are product stats, so the partial exact layer is already exact
    if n != n_exact:
        ising.diagonalise()
    #print(np.shape(wf))
    #print(np.shape(ising.eig_vec[:,0]))
    #print("wf: \n {}\nising.eig_vec[:,0]: \n {}".format(wf, np.round(ising.eig_vec[:,0], decimals=6)))
    cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(wf, ising.eig_vec[:,0], rtol=1e-7, atol=1e-8)

@pytest.mark.parametrize(
    "n, subsystem_qubits",
    [
        (
            [1, 2],
            None
        ),
        (
            [2, 1],
            None
        ),
        (
            [2, 2],
            None
        ),
        (
            [1, 3],
            None 
        ),
        (
            [1, 2],
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)]]
        ),

        (
            [2, 1],
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)]]
        ),
        (
            [2, 2],
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(1,0), cirq.GridQubit(1,1)]]
        ),
        (
            [1, 3],
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(0,2)]]  
        ),
    ]
)
def test__exact_layer_cc(n,subsystem_qubits):
    print(n)
    j_v = 2*(np.random.rand(n[0]-1,n[1])- 0.5)
    j_h = 2*(np.random.rand(n[0],n[1]-1)- 0.5)
    h = 2*(np.random.rand(n[0],n[1])- 0.5)
    print("j_v: {}\nj_h {}\nh {}".format(j_v, j_h, h))
    ising= Ising("GridQubit", n, j_v, j_h, h, "X")

    ising.set_simulator("cirq")
    ising.set_circuit("basics",{    "start": "exact", 
                                    "n_exact": n,
                                    "subsystem_qubits": subsystem_qubits,
                                    "cc_exact": True})

    #Starting at the nth Ising eigenvector and applying U^-1 = U^dagger
    #Should end up in the nth Z-Eigenstate
    wf0 = np.zeros(2**(n[0]*n[1]))

    for i in range(2**(n[0]*n[1])):
        wf = ising.simulator.simulate(  initial_state=ising.eig_vec[:,i],
                                        program=ising.circuit).state_vector()
        wf = wf/np.linalg.norm(wf)

        wf0[i]=1
        wf0[i-1]=0

        print("i: {}\nwf: {}\nwf0: {}".format(i,wf, wf0))
        cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(wf, wf0, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize(
    "n, n_exact, subsystem_qubits",
    [
        (
            [2, 2], 
            [2, 1],
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
            [cirq.GridQubit(0,1), cirq.GridQubit(1,1)]]
        ),
        (
            [2, 2], 
            [2, 2],
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(1,0), cirq.GridQubit(1,1)]]
        ),
        (
            [2, 2], 
            [1, 2],
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
            [cirq.GridQubit(1,0), cirq.GridQubit(1,1)]]
        ),
        (
            [2, 3], 
            [2, 1],
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
            [cirq.GridQubit(0,1), cirq.GridQubit(1,1)],
            [cirq.GridQubit(0,2), cirq.GridQubit(1,2)]]
        ),
        (
            [2, 3], 
            [1, 3],
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1),  cirq.GridQubit(0,2)],
            [cirq.GridQubit(1,0), cirq.GridQubit(1,1), cirq.GridQubit(1,2)]]
        ),
        (
            [1, 6], 
            [1, 3],
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1),  cirq.GridQubit(0,2)],
            [cirq.GridQubit(0,3), cirq.GridQubit(0,4), cirq.GridQubit(0,5)]]
        ),
    ]
)
def test__exact_layer_subsystem_qubits(n,n_exact,subsystem_qubits):
    j_v = 2*(np.random.rand(n[0]-1,n[1])- 0.5)
    j_h = 2*(np.random.rand(n[0],n[1]-1)- 0.5)
    h = 2*(np.random.rand(n[0],n[1])- 0.5)
    print("j_v: {}\nj_h {}\nh {}".format(j_v, j_h, h))
    ising= Ising("GridQubit", n, j_v, j_h, h, "X")
    ising.set_circuit("basics",{    "start": "exact", 
                                    "n_exact": n_exact})

    print("ising.circuit:\n{}\n".format(ising.circuit))
    ising2= Ising("GridQubit", n, j_v, j_h, h, "X")
    ising2.set_circuit("basics",{    "start": "exact", 
                                    "subsystem_qubits": subsystem_qubits})

    print("ising2.circuit:\n{}\n".format(ising2.circuit))

    assert(ising.circuit == ising2.circuit)

@pytest.mark.parametrize(
    "n, n_exact, j_v, j_h, h, subsystem_qubits, subsystem_h",
    [
        (
            [2, 2], 
            [2, 1],
            2*(np.random.rand(2-1,2)- 0.5),
            2*(np.random.rand(2,2-1)- 0.5),
            np.ones((2,2)),
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
            [cirq.GridQubit(0,1), cirq.GridQubit(1,1)]],
            [    np.transpose(np.array([ [[1], [1]]]), (1,0, 2)), 
                np.transpose(np.array([ [[1], [1]] ]), (1,0, 2))] ,      
        ),
        (
            [2, 2], 
            [1, 2],
            2*(np.random.rand(2-1,2)- 0.5),
            2*(np.random.rand(2,2-1)- 0.5),
            np.ones((2,2)),
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
            [cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
            [    np.transpose(np.array([ [[1], [1]]]), (0,1, 2)), 
                np.transpose(np.array([ [[1], [1]] ]), (0,1, 2))] ,      
        ),
        (
            [2, 3], 
            [1, 3],
            2*(np.random.rand(2-1,3)- 0.5),
            2*(np.random.rand(2,3-1)- 0.5),
            np.ones((2,3)),
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(0,2)],
            [cirq.GridQubit(1,0), cirq.GridQubit(1,1), cirq.GridQubit(1,2)]],
            [    np.transpose(np.array([ [[1], [1], [1]]]), (0,1, 2)), 
                np.transpose(np.array([ [[1], [1], [1]] ]), (0,1, 2))] ,      
        ),
        (
            [2, 3], 
            [1, 3],
            2*(np.random.rand(2-1,3)- 0.5),
            2*(np.random.rand(2,3-1)- 0.5),
            np.arange(1,1+6).reshape((2, 3)),
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(0,2)],
            [cirq.GridQubit(1,0), cirq.GridQubit(1,1), cirq.GridQubit(1,2)]],
            [    np.transpose(np.array([ [[1], [2], [3]]]), (0,1, 2)), 
                np.transpose(np.array([ [[4], [5], [6]] ]), (0,1, 2))] ,      
        ),
        (
            [2, 2], 
            [2, 1],
            np.zeros((2-1,2)),
            np.zeros((2,2-1)),
            np.ones((2,2)),
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
            [cirq.GridQubit(0,1), cirq.GridQubit(1,1)]],
            [    np.transpose(np.array([ [[1], [1]]]), (1,0, 2)), 
                np.transpose(np.array([ [[1], [1]] ]), (1,0, 2))] ,      
        ),
        (
            [1, 2], 
            [1, 2],
            np.zeros((1-1,2)),
            np.zeros((1,2-1)),
            np.arange(2).reshape((1, 2)),
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)]],
            [    np.transpose(np.array([ [[0], [1]]]), (0,1, 2))] ,      
        ),
        (
            [2, 1], 
            [2, 1],
            np.zeros((2-1,1)),
            np.zeros((2,1-1)),
            np.arange(2).reshape((2, 1)),
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)]],
            [    np.transpose(np.array([ [[0], [1]]]), (1,0, 2))] ,      
        ),
        (
            [2, 2], 
            [2, 1],
            np.zeros((2-1,2)),
            np.zeros((2,2-1)),
            np.array([0,2,1,3]).reshape((2, 2)),
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
            [cirq.GridQubit(0,1), cirq.GridQubit(1,1)]],
            [    np.transpose(np.array([ [[0], [1]]]), (1,0, 2)), 
                np.transpose(np.array([ [[2], [3]] ]), (1,0, 2))] ,      
        ),
        (
            [2, 2], 
            [2, 1],
            np.zeros((2-1,2)),
            np.zeros((2,2-1)),
            np.array([0,1,0,1]).reshape((2, 2)),
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
            [cirq.GridQubit(0,1), cirq.GridQubit(1,1)]],
            [    np.transpose(np.array([ [[0], [0]]]), (1,0, 2)), 
                np.transpose(np.array([ [[1], [1]] ]), (1,0, 2))] ,      
        ),
        #(
        #    [2, 4], 
        #    [2, 2],
        #    np.array([0,1,0,1]).reshape((2-1, 4)),
        #    np.array([0,1,0,1]).reshape((2, 4-1)),
        #    np.array([0,1,0,1]).reshape((2, 4)),
        #    [[cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(0,1), cirq.GridQubit(1,1)],
        #    [cirq.GridQubit(2,0), cirq.GridQubit(3,0), cirq.GridQubit(2,1), cirq.GridQubit(3,1)]],
        #    [    np.transpose(np.array([ [[0], [0]]]), (1,0, 2)), 
        #        np.transpose(np.array([ [[1], [1]] ]), (1,0, 2))] ,      
        #),
    ]
)
def test__exact_layer_subsystem_h(n,n_exact,j_v, j_h, h, subsystem_qubits, subsystem_h):
    print("j_v: {}\nj_h {}\nh {}".format(j_v, j_h, h))
    ising= Ising("GridQubit", n, j_v, j_h, h, "X")
    ising.set_circuit("basics",{   "start": "exact", 
                                    "n_exact": n_exact})

    print("ising.circuit:\n{}\n".format(ising.circuit))
    h0 = 2*(np.random.rand(n[0],n[1])- 0.5)
    ising2 = Ising("GridQubit", n, j_v, j_h, h0, "X")
    ising2.set_circuit("basics",{    "start": "exact", 
                                    "subsystem_qubits": subsystem_qubits,
                                    "subsystem_h": subsystem_h})
    print("ising2.circuit:\n{}\n".format(ising2.circuit))

    #Only compare unitaries if circuits do not correspond
    if ising.circuit == ising2.circuit:
        assert True
    else:
        print("ising.circuit.all_qubits():\n{}\nising2.circuit.all_qubits():\n{}\n".format(ising.circuit.all_qubits(),ising2.circuit.all_qubits()))
        print("ising.circuit.unitary()-ising2.circuit.unitary()\n{}\n".format(ising.circuit.unitary()-ising2.circuit.unitary()))
        cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(ising.circuit.unitary(),ising2.circuit.unitary(), rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize(
    "n, n_exact, j_v, j_h, h, subsystem_qubits, subsystem_j_v, subsystem_j_h",
    [
        (
            [2, 1], 
            [2, 1],
            np.ones((2-1,1)),
            np.ones((2,1-1)),
            np.zeros((2,1)),
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)]],
            [    np.transpose(np.array([ [[1]]]), (1,0, 2)) ]  ,     
            [np.array([]).reshape(2, 0, 1)]
        ),
        (
            [2, 1], 
            [2, 1],
            np.ones((2-1,1)),
            np.ones((2,1-1)),
            2*(np.random.rand(2,1)- 0.5),
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)]],
            [    np.transpose(np.array([ [[1]]]), (1,0, 2)) ]  ,     
            [np.array([]).reshape(2, 0, 1)]  
        ),
        (
            [1, 2], 
            [1, 2],
            np.ones((1-1,2)),
            np.ones((1,2-1)),
            np.zeros((1,2)),
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)]],
            [np.array([]).reshape(0, 2, 1)],
            [    np.transpose(np.array([ [[1]]]), (1,0, 2)) ]   
        ),
        (
            [1, 2], 
            [1, 2],
            np.ones((1-1,2)),
            np.ones((1,2-1)),
            2*(np.random.rand(1,2)- 0.5),
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)]],
            [np.array([]).reshape(0, 2, 1)],
            [    np.transpose(np.array([ [[1]]]), (1,0, 2)) ]   
        ),
        (
            [2, 2], 
            [2, 1],
            np.ones((2-1,2)),
            np.ones((2,2-1)),
            np.zeros((2,2)),
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
            [cirq.GridQubit(0,1), cirq.GridQubit(1,1)]],
            [   np.transpose(np.array([ [[1]]]), (1,0, 2)),
                np.transpose(np.array([ [[1]]]), (1,0, 2))],     
            [   np.array([]).reshape(2, 0, 1),
                np.array([]).reshape(2, 0, 1)]
        ),
        (
            [2, 2], 
            [1, 2],
            np.ones((2-1,2)),
            np.ones((2,2-1)),
            np.zeros((2,2)),
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
            [cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
            [   np.array([]).reshape(0, 2, 1),
                np.array([]).reshape(0, 2, 1)],
            [   np.transpose(np.array([ [[1]]]), (1,0, 2)),
                np.transpose(np.array([ [[1]]]), (1,0, 2))]
        ),
        (
            [2, 1], 
            [2, 1],
            np.zeros((2-1,1)),
            np.zeros((2,1-1)),
            2*(np.random.rand(2,1)- 0.5),
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)]],
            [   np.transpose(np.array([ [[0]]]), (1,0, 2))],     
            [   np.array([]).reshape(2, 0, 1)]
        ),
        (
            [2, 2], 
            [2, 1],
            np.zeros((2-1,2)),
            np.zeros((2,2-1)),
            2*(np.random.rand(2,2)- 0.5),
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
            [cirq.GridQubit(0,1), cirq.GridQubit(1,1)]],
            [   np.transpose(np.array([ [[0]]]), (1,0, 2)),
                np.transpose(np.array([ [[0]]]), (1,0, 2))],
            [   np.array([]).reshape(2, 0, 1),
                np.array([]).reshape(2, 0, 1)]
        ),
        (
            [2, 2], 
            [2, 1],
            np.ones((2-1,2)),
            np.zeros((2,2-1)),
            2*(np.random.rand(2,2)- 0.5),
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
            [cirq.GridQubit(0,1), cirq.GridQubit(1,1)]],
            [   np.transpose(np.array([ [[1]]]), (1,0, 2)),
                np.transpose(np.array([ [[1]]]), (1,0, 2))],
            [   np.array([]).reshape(2, 0, 1),
                np.array([]).reshape(2, 0, 1)]
        ),
        (
            [2, 2], 
            [1, 2],
            np.zeros((2-1,2)),
            np.ones((2,2-1)),
            2*(np.random.rand(2,2)- 0.5),
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
            [cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
            [   np.array([]).reshape(0, 2, 1),
                np.array([]).reshape(0, 2, 1)],
            [   np.transpose(np.array([ [[1]]]), (1,0, 2)),
                np.transpose(np.array([ [[1]]]), (1,0, 2))]
        ),
        (
            [2, 3], 
            [2, 1],
            np.ones((2-1,3)),
            np.zeros((2,3-1)),
            2*(np.random.rand(2,3)- 0.5),
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
            [cirq.GridQubit(0,1), cirq.GridQubit(1,1)],
            [cirq.GridQubit(0,2), cirq.GridQubit(1,2)]],
            [   np.transpose(np.array([ [[1]]]), (1,0, 2)),
                np.transpose(np.array([ [[1]]]), (1,0, 2)),
                np.transpose(np.array([ [[1]]]), (1,0, 2))],
            [   np.array([]).reshape(2, 0, 1),
                np.array([]).reshape(2, 0, 1),
                np.array([]).reshape(2, 0, 1)]
        ),
        (
            [1, 3], 
            [1, 3],
            np.zeros((1-1,3)),
            np.ones((1,3-1)),
            np.zeros((1,3)),
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(0,2)]],
            [   np.array([]).reshape(0, 3, 1)],
            [   np.transpose(np.array([ [[1]], [[1]]]), (1,0, 2))]
        ),
        (
            [1, 3], 
            [1, 3],
            np.zeros((1-1,3)),
            np.ones((1,3-1)),
            2*(np.random.rand(1,3)- 0.5),
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(0,2)]],
            [   np.array([]).reshape(0, 3, 1)],
            [   np.transpose(np.array([ [[1]], [[1]]]), (1,0, 2))]
        ),
        (
            [2, 3], 
            [1, 3],
            np.zeros((2-1,3)),
            np.ones((2,3-1)),
            2*(np.random.rand(2,3)- 0.5),
            [[cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(0,2)],
            [cirq.GridQubit(1,0), cirq.GridQubit(1,1), cirq.GridQubit(1,2)]],
            [   np.array([]).reshape(0, 3, 1),
                np.array([]).reshape(0, 3, 1)],
            [   np.transpose(np.array([ [[1]], [[1]]]), (1,0, 2)),
                np.transpose(np.array([ [[1]], [[1]]]), (1,0, 2))]
        ),
        (
            [3, 2], 
            [3, 1],
            np.ones((3-1,2)),
            np.zeros((3,2-1)),
            2*(np.random.rand(3,2)- 0.5),
            [[cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(2,0)],
            [cirq.GridQubit(0,1), cirq.GridQubit(1,1), cirq.GridQubit(2,1)]],
            [   np.transpose(np.array([ [[1], [1]]]), (1,0, 2)),
                np.transpose(np.array([ [[1], [1]]]), (1,0, 2))],
            [   np.array([]).reshape(3, 0, 1),
                np.array([]).reshape(3, 0, 1)]
        ),
    ]
)
def test__exact_layer_subsystem_J(n,n_exact,j_v, j_h, h, subsystem_qubits, subsystem_j_v, subsystem_j_h):
    print("j_v: {}\nj_h {}\nh {}".format(j_v, j_h, h))
    ising= Ising("GridQubit", n, j_v, j_h, h, "X")
    ising.set_circuit("basics",{   "start": "exact", 
                                    "n_exact": n_exact})

    print("ising.circuit:\n{}\n".format(ising.circuit))

    print("subsystem_j_v:\n{}\nsubsystem_j_h:\n{}".format(subsystem_j_v, subsystem_j_h))
    j_v0 = 2*(np.random.rand(n[0]-1,n[1])- 0.5)
    j_h0 = 2*(np.random.rand(n[0],n[1]-1)- 0.5)
    ising2 = Ising("GridQubit", n, j_v0, j_h0, h, "X")
    ising2.set_circuit("basics",{    "start": "exact", 
                                    "subsystem_qubits": subsystem_qubits,
                                    "subsystem_j_h": subsystem_j_h,
                                    "subsystem_j_v": subsystem_j_v})
    print("ising2.circuit:\n{}\n".format(ising2.circuit))

    #Only compare unitaries if circuits do not correspond
    if ising.circuit == ising2.circuit:
        assert True
    else:
        print("ising.circuit.all_qubits():\n{}\nising2.circuit.all_qubits():\n{}\n".format(ising.circuit.all_qubits(),ising2.circuit.all_qubits()))
        print("ising.circuit.unitary()-ising2.circuit.unitary()\n{}\n".format(ising.circuit.unitary()-ising2.circuit.unitary()))
        cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(ising.circuit.unitary(),ising2.circuit.unitary(), rtol=1e-15, atol=1e-15)

@pytest.mark.parametrize(
    "n,SingleQubitGates, TwoQubitGates",
    [
        (
            [1, 2],
            [[cirq.X]],
            [[lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)]]
            ),
            (
            [2, 2],
            [[cirq.X]],
            [[lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)]]
            ),
    ]
)
def test__exact_layer_QubitGates(n,SingleQubitGates, TwoQubitGates):
    j_v0 = 2*(np.random.rand(n[0]-1,n[1])- 0.5)
    j_h0 = 2*(np.random.rand(n[0],n[1]-1)- 0.5)
    h0 = 2*(np.random.rand(n[0],n[1])- 0.5)
    ising = Ising("GridQubit", n, j_v0, j_h0, h0, "X")
    ising.set_circuit("basics",{    "start": "exact",
                                    "b_exact": None,
                                    "n_exact": n})
    print("ising.circuit:\n{}\n".format(ising.circuit))

    spinmodel = SpinModel("GridQubit", 
                                        n, 
                                        [j_v0], 
                                        [j_h0], 
                                        [h0],
                                        [lambda q1, q2: cirq.Y(q1)*cirq.Y(q2)],
                                        [cirq.Y])
    spinmodel.set_circuit("basics",{    "start": "exact",
                                    "n_exact": n,
                                    "b_exact": [1,1],
                                    "SingleQubitGates": SingleQubitGates,
                                    "TwoQubitGates": TwoQubitGates})

    print("spinmodel.circuit:\n{}\n".format(spinmodel.circuit))

    #Only compare unitaries if circuits do not correspond
    if ising.circuit == spinmodel.circuit:
        assert True
    else:
        print("ising.circuit.all_qubits():\n{}\nspinmodel.circuit.all_qubits():\n{}\n".format(ising.circuit.all_qubits(),spinmodel.circuit.all_qubits()))
        print("ising.circuit.unitary()-spinmodel.circuit.unitary()\n{}\n".format(ising.circuit.unitary()-spinmodel.circuit.unitary()))
        cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(ising.circuit.unitary(),spinmodel.circuit.unitary(), rtol=1e-15, atol=1e-15)
    

def test__identity_layer():
    ising= Ising("GridQubit", [2, 2], np.ones((1, 2)), np.ones((2, 1)), np.ones((2, 2)))

    test_circuit = cirq.Circuit()
    for qubit in list(itertools.chain(*ising.qubits)):
        test_circuit.append(cirq.I.on(qubit))

    ising.set_circuit("basics",{   "start": "identity",})
    assert test_circuit == ising.circuit

    ising.set_circuit("basics",{   "append": False, "end": "identity"})
    assert test_circuit == ising.circuit

def test__hadamard_layer():
    ising= Ising("GridQubit", [2, 2], np.ones((1, 2)), np.ones((2, 1)), np.ones((2, 2)))
    ising.set_circuit("basics",{   "start": "hadamard",})

    test_circuit = cirq.Circuit()
    for qubit in list(itertools.chain(*ising.qubits)):
        test_circuit.append(cirq.H.on(qubit))
    assert test_circuit == ising.circuit

    ising.set_circuit("basics",{   "append": False, "end": "hadamard"})
    assert test_circuit == ising.circuit

@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, field, theta, location",
    [
        ("GridQubit",
            [2, 2],
            np.ones((1, 2)) / 2,
            np.ones((2, 1)) / 5,
            np.ones((2, 2)),
            "X",
            1/2,
            "start"),
        ("GridQubit",
            [2, 2],
            -np.ones((1, 2)) / 2,
            -np.ones((2, 1)) / 5,
            np.ones((2, 2)),
            "X",
            1/2,
            "end"),
        ("GridQubit",
            [2, 2],
            np.ones((1, 2)) / 2,
            np.ones((2, 1)) / 5,
            np.zeros((2, 2)) / 3,
            "X",
            0,
            "start"),
        ("GridQubit",
            [2, 2],
            np.ones((1, 2)),
            np.ones((2, 1)),
            np.ones((2, 2))/2,
            "X",
            1/6,
            "start"),
        ("GridQubit",
            [2, 2],
            np.ones((1, 2)),
            np.ones((2, 1)),
            np.ones((2, 2))/np.sqrt(2),
            "X",
            1/4,
            "start"),
        ("GridQubit",
            [2, 3],
            np.ones((2, 3)),
            np.ones((2, 2)),
            np.ones((2, 3))/2,
            "X",
            1/6,
            "start"),
    ]
    )
def test__mf_layer(qubittype, n, j_v, j_h, h, field, theta, location):
    ising= Ising(qubittype, n, j_v, j_h, h, field)
    ising.set_circuit("basics", {location: "mf"})
    print("ising.circuit: \n{}\n".format(ising.circuit))    
    
    test_circuit=cirq.Circuit()
    for qubit in list(itertools.chain(*ising.qubits)):
        if abs(theta) < 0.5 and j_v[0][0] > 0:
            sign = (-1)**(qubit._col+qubit._row)
        else:
            sign=1
        test_circuit.append(cirq.XPowGate(exponent=(sign*theta)).on(qubit) )

    print("test_circuit: \n{}\n".format(test_circuit))    
    
    #This is not very stable
    #print(test_circuit.unitary(), ising.circuit.unitary())
    #assert test_circuit == ising.circuit

    #Better:
    cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(test_circuit.unitary(),ising.circuit.unitary(), rtol=1e-15, atol=1e-15)


def test__neel_layer():
    ising= Ising("GridQubit", [2, 3], np.ones((1, 3)), np.ones((2, 2)), np.ones((2, 3)))
    ising.set_circuit("basics",{   "start": "neel",})

    test_circuit = cirq.Circuit()
    
    test_circuit.append(cirq.X.on(cirq.GridQubit(0,0)))
    test_circuit.append(cirq.X.on(cirq.GridQubit(0,2)))
    test_circuit.append(cirq.X.on(cirq.GridQubit(1,1)))

    assert test_circuit == ising.circuit

    ising.set_circuit("basics",{   "append": False, "end": "neel"})
    assert test_circuit == ising.circuit

def test_add_missing_cpv():
    ising= Ising("GridQubit", [1, 3], np.ones((0, 3)), np.ones((1, 2)), np.ones((1, 3)))
    ising.set_circuit("hea", {"p": 5, "parametirsation": "individual"})
    ising.set_circuit("basics", {"start": "hadamard"})
    
    init_circuit_param = ising.circuit_param.copy()
    for i in range(np.size(ising.circuit_param)-1,0,-1):
        if np.random.random() > 0.5:
            del ising.circuit_param[i]
    ising.circuit_param_values = np.zeros(np.size(ising.circuit_param))
    
    print(ising.circuit_param)
    ising.basics.add_missing_cpv(ising)

    print(ising.circuit_param)
    assert np.size(init_circuit_param) == np.size(ising.circuit_param)
    assert set(init_circuit_param) == set(ising.circuit_param)
    assert np.size(ising.circuit_param) == np.size(ising.circuit_param_values)

def test_rm_unused_cpv():
    ising= Ising("GridQubit", [1, 3], np.ones((0, 3)), np.ones((1, 2)), np.ones((1, 3)))
    ising.set_circuit("hea", {"p": 5, "parametirsation": "individual"})
    ising.set_circuit("basics", {"start": "hadamard"})
    
    init_circuit_param = ising.circuit_param.copy()
    ising.circuit_param_values = np.arange(np.size(ising.circuit_param))
    init_circuit_param_values = ising.circuit_param_values.copy()

    for i in range(np.size(ising.circuit_param)-1,0,-1):
        if np.random.random() > 0.5:
            ising.circuit_param.insert(i, sympy.Symbol("test" + str(i)))
            ising.circuit_param_values = np.insert(ising.circuit_param_values, i, -1, axis = 0 )
    
    #print(ising.circuit_param)
    #print(ising.circuit_param_values)
    
    ising.basics.rm_unused_cpv(ising)

    #print(ising.circuit_param)
    #print(ising.circuit_param_values)

    assert np.size(init_circuit_param) == np.size(ising.circuit_param)
    assert set(init_circuit_param) == set(ising.circuit_param)
    assert np.size(ising.circuit_param) == np.size(ising.circuit_param_values)
    assert (init_circuit_param_values == ising.circuit_param_values ).all

def test_set_circuit_errors():
    ising= Ising("GridQubit", [1, 3], np.ones((0, 3)), np.ones((1, 2)), np.ones((1, 3)), "Z")
    with pytest.raises(AssertionError):
        ising.set_circuit("basics", {"start": "test"})

    with pytest.raises(AssertionError):
        ising.set_circuit("basics", {"end": "test"})

    with pytest.raises(AssertionError):
        ising.set_circuit("basics", {"start": "mf"})

    with pytest.warns(UserWarning):
        ising.set_circuit("basics", {"start": "exact"})

    ising.qubittype = "test"
    with pytest.raises(NotImplementedError):
        ising.set_circuit("basics", {"start": "exact"})