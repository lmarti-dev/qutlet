# external imports
from unicodedata import decimal
import pytest
import numpy as np
import cirq
import sympy
import itertools

# internal imports
from fauvqe import Ising, SpinModel, Converter
from fauvqe.models.circuits.basics import SpinModelDummy
from fauvqe.objectives.abstractexpectationvalue import AbstractExpectationValue
from fauvqe.objectives.expectationvalue import ExpectationValue
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

    ising.set_simulator("cirq", {"dtype": np.complex128})
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
        cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(wf, wf0, rtol=1e-14, atol=5e-14)

@pytest.mark.higheffort
@pytest.mark.parametrize(
    "n, basics_options",
    [
        #Here the "subsystem" is already the entire system
        #These are included here as less convoluted tests
        #Having them pass but the real subsystem fail helps 
        (
            [1, 2],
            {"subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)]]}
        ),
        #Mix qubit order to get non-standard one
        (
            [2, 2],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,1),
                                    cirq.GridQubit(1,0), cirq.GridQubit(0,1)]]}
        ),
        #Test [2,4] block as largest block in real sim
        (
            [2, 4],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(0,1), cirq.GridQubit(1,1),
                                    cirq.GridQubit(0,2), cirq.GridQubit(1,2), cirq.GridQubit(0,3), cirq.GridQubit(1,3)]]}
        ),
        #Below the "subsystem" is already the entire system
        # 2 subsystems
        (
            [2, 2],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
                                    [cirq.GridQubit(1,0), cirq.GridQubit(1,1)]]}
        ),
        (
            [2, 2],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
                                    [cirq.GridQubit(0,1), cirq.GridQubit(1,1)]]}
        ),
        # 3 and more subsystems
        (
            [2, 3],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
                                    [cirq.GridQubit(0,1), cirq.GridQubit(1,1)],
                                    [cirq.GridQubit(0,2), cirq.GridQubit(1,2)]]}
        ),
        (
            [3, 2],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
                                    [cirq.GridQubit(1,0), cirq.GridQubit(1,1)],
                                    [cirq.GridQubit(2,0), cirq.GridQubit(2,1)]]}
        ),
        #Mix up subsystem qubit order
        #Mixing subsystems work
        (
            [3, 2],
            {"subsystem_qubits": [  [cirq.GridQubit(1,0), cirq.GridQubit(1,1)],
                                    [cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
                                    [cirq.GridQubit(2,0), cirq.GridQubit(2,1)]]}
        ),
        #Mixing qubit order within the subsystems still fails:
        #(
        #    [3, 2],
        #    {"subsystem_qubits": [  [cirq.GridQubit(1,0), cirq.GridQubit(1,1)],
        #                            [cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
        #                            [cirq.GridQubit(2,1), cirq.GridQubit(2,0)]]}
        #),
        (
            [2, 4],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
                                    [cirq.GridQubit(0,1), cirq.GridQubit(0,2), cirq.GridQubit(1,1), cirq.GridQubit(1,2)],
                                    [cirq.GridQubit(0,3), cirq.GridQubit(1,3)]]}
        ),
        (
            [4, 2],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
                                    [cirq.GridQubit(1,0), cirq.GridQubit(1,1), cirq.GridQubit(2,0), cirq.GridQubit(2,1)],
                                    [cirq.GridQubit(3,0), cirq.GridQubit(3,1)]]}
        ),
        (
            [3, 3],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(2,0)],
                                    [cirq.GridQubit(0,1), cirq.GridQubit(1,1), cirq.GridQubit(2,1)],
                                    [cirq.GridQubit(0,2), cirq.GridQubit(1,2), cirq.GridQubit(2,2)]]}
        ),
        # This takes long and fails
        #(
        #    [3, 4],
        #    {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(2,0)],
        #                            [cirq.GridQubit(0,1), cirq.GridQubit(1,1), cirq.GridQubit(2,1),
        #                            cirq.GridQubit(0,2), cirq.GridQubit(1,2), cirq.GridQubit(2,2)],
        #                            [cirq.GridQubit(0,3), cirq.GridQubit(1,3), cirq.GridQubit(2,3)]]}
        #),
        #This works but takes long
        (
            [3, 4],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(2,0)],
                                    [cirq.GridQubit(0,1), cirq.GridQubit(1,1), cirq.GridQubit(2,1)],
                                    [cirq.GridQubit(0,2), cirq.GridQubit(1,2), cirq.GridQubit(2,2)],
                                    [cirq.GridQubit(0,3), cirq.GridQubit(1,3), cirq.GridQubit(2,3)]]}
        ),
        #Test coverings like used in later simulations:
        #1x4, 4x1 covering
        (
            [4, 4],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(2,0), cirq.GridQubit(3,0)],
                                    [cirq.GridQubit(0,1), cirq.GridQubit(1,1), cirq.GridQubit(2,1), cirq.GridQubit(3,1)],
                                    [cirq.GridQubit(0,2), cirq.GridQubit(1,2), cirq.GridQubit(2,2), cirq.GridQubit(3,2)],
                                    [cirq.GridQubit(0,3), cirq.GridQubit(1,3), cirq.GridQubit(2,3), cirq.GridQubit(3,3)]]}
        ),
        (
            [4, 4],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(0,2), cirq.GridQubit(0,3)],
                                    [cirq.GridQubit(1,0), cirq.GridQubit(1,1), cirq.GridQubit(1,2), cirq.GridQubit(1,3)],
                                    [cirq.GridQubit(2,0), cirq.GridQubit(2,1), cirq.GridQubit(2,2), cirq.GridQubit(2,3)],
                                    [cirq.GridQubit(3,0), cirq.GridQubit(3,1), cirq.GridQubit(3,2), cirq.GridQubit(3,3)]]}
        ),
        #2x4, 1x4_2x4_1x4 covering
        (
            [4, 4],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(0,2), cirq.GridQubit(0,3),
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1), cirq.GridQubit(1,2), cirq.GridQubit(1,3)],
                                    [cirq.GridQubit(2,0), cirq.GridQubit(2,1), cirq.GridQubit(2,2), cirq.GridQubit(2,3),
                                    cirq.GridQubit(3,0), cirq.GridQubit(3,1), cirq.GridQubit(3,2), cirq.GridQubit(3,3)]]}
        ),
        (
            [4, 4],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(0,2), cirq.GridQubit(0,3)],
                                    [cirq.GridQubit(1,0), cirq.GridQubit(1,1), cirq.GridQubit(1,2), cirq.GridQubit(1,3),
                                    cirq.GridQubit(2,0), cirq.GridQubit(2,1), cirq.GridQubit(2,2), cirq.GridQubit(2,3)],
                                    [cirq.GridQubit(3,0), cirq.GridQubit(3,1), cirq.GridQubit(3,2), cirq.GridQubit(3,3)]]}
        ),
    ]
)
def test_subsystem_U(n, basics_options):
    #Take random h \in 0.1 ..1 and J \in +- 0.1 .. 1
    #Take random SingleQubiteGate and random TwoQubiteGate
    #Check whether
    # U^\dagger |\phi_m^(HA)> \otimes |\phi_l^(HB)> = |m> \otimes |l>
    j_v = 2*0.9*(np.random.rand(n[0]-1,n[1])- 0.5)
    j_v = j_v + np.sign(j_v)+0.1
    j_h = 2*0.9*(np.random.rand(n[0],n[1]-1)- 0.5)
    j_v = j_v + np.sign(j_v)+0.1
    h = 0.9*(np.random.rand(n[0],n[1])+ 0.1)
    #print("j_v: {}\nj_h {}\nh {}".format(j_v, j_h, h))

    #This does not actually matter as Ising itself is not used
    ising= Ising("GridQubit", n, j_v, j_h, h, "X")
    ising.set_simulator("cirq", {"dtype": np.complex128})

    #Get random SingleQubiteGate and random TwoQubiteGate 
    #[[cirq.X]],
    #[[lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)]]
    #_pauligates = [cirq.I, cirq.X, cirq.Y, cirq.Z]

    _pauligates = [cirq.X, cirq.Y, cirq.Z]
    tmp_SQG = _pauligates[np.random.randint(3)]
    tmp_TQG = lambda q1, q2: _pauligates[np.random.randint(3)](q1)*_pauligates[np.random.randint(3)](q2)

    common_options = {    "start": "exact", 
                          "append": False,
                          "cc_exact": True,
                          "SingleQubitGates": [[tmp_SQG]],
                          "TwoQubitGates": [[tmp_TQG]]}
    common_options.update(basics_options)
    ising.set_circuit("basics", common_options)
    _qubit_order = ising.basics.get_subsystem_qubit_map(ising)



    #Prepare energy comparison
    AExpValue_obj = AbstractExpectationValue(ising,
                                            sum(ising.subsystem_hamiltonians))
    _energy_filter = ising.basics.get_energy_filter_from_subsystem(ising)
    
    #Starting at the nth Ising eigenvector and applying U^-1 = U^dagger
    #Should end up in the nth Z-Eigenstate
    wf0 = np.zeros(2**(n[0]*n[1]))

    #For less than 8 qubits check every basis vector
    #Otherwise check a random sample
    if n[0]*n[1] < 9:
        indices = range(2**(n[0]*n[1]))
    else:
        rng = np.random.default_rng()
        indices=rng.integers(low=0, high=2**(n[0]*n[1]), size=16)

    previous_i = 0
    for i in indices:
        #Get composite subststem eigenstate by tensorproduct
        if len(ising.subsystem_qubits) == 1:
            in_state = ising.eig_vec[:,i]
            E_filter= _energy_filter[i]
        else:
            # Maybe this is a function that also should be provided in circuits.basics
            # i to binary
            # cut subsystems accordingly
            # retransform to int to use in ising.subsystem_U[0][:,3] etc

            tmp_binary = np.binary_repr(i, width=n[0]*n[1])

            i_sub_0 = int(tmp_binary[:len(ising.subsystem_qubits[0])],2)
            tmp_binary = tmp_binary[len(ising.subsystem_qubits[0]):]
            #print(i_sub_0, tmp_binary)

            i_sub_1 = int(tmp_binary[:len(ising.subsystem_qubits[1])],2)
            tmp_binary = tmp_binary[len(ising.subsystem_qubits[1]):]
            #print(i_sub_1, tmp_binary)

            in_state= np.tensordot( ising.subsystem_U[0][:,i_sub_0], 
                                    ising.subsystem_U[1][:,i_sub_1], 
                                    axes=0).reshape(2**(len(ising.subsystem_qubits[0])+len(ising.subsystem_qubits[1])))

            #To do add code for len(ising.subsystem_qubits) > 2
            for n_sub in range(2, len(ising.subsystem_qubits)):
                i_sub = int(tmp_binary[:len(ising.subsystem_qubits[n_sub])],2)
                tmp_binary = tmp_binary[len(ising.subsystem_qubits[n_sub]):]

                in_state = np.tensordot( in_state, 
                                        ising.subsystem_U[n_sub][:,i_sub], 
                                        axes=0).reshape(len(in_state)*2**(len(ising.subsystem_qubits[n_sub])))
            
            #in_state= np.tensordot(ising.subsystem_U[0][:,3], ising.subsystem_U[1][:,3], axes=0).reshape(2**(n[0]*n[1]))
        
        #This tests U^\dagger |\phi_m^(HA)> \otimes |\phi_l^(HB)> = |m> \otimes |l>
        wf = ising.simulator.simulate(  initial_state=in_state,
                                        program=ising.circuit,
                                        qubit_order=_qubit_order).state_vector()
        wf = wf/np.linalg.norm(wf)


        wf0[previous_i]=0
        previous_i = i

        wf0[i]=1
        

        #print("i: {}\nwf: {}\nwf0: {}".format(i,wf, wf0))
        cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(wf, wf0, rtol=n[0]*n[1]*1e-14, atol=n[0]*n[1]*5e-14)
        
        #To Do also check whether energy in energy filter is correct
        # Once calc energy via AbstractExpectationValue, once use index in energy filter
        
        #use here qubit order for subsystem = system
        if len(ising.subsystem_qubits) == 1:
            E_AEV = AExpValue_obj.evaluate( atol=1e-14,
                                            wavefunction=in_state)/(n[0]*n[1])
        else:
            E_AEV = AExpValue_obj.evaluate( atol=1e-14,
                                            q_map=_qubit_order,
                                            wavefunction=in_state)/(n[0]*n[1])
        #print("Energy from AbstractExpectationValue: \t{}\nEnergy from energy filter: \t\t{}\nDifference: \t\t\t\t {}".format(E_AEV,_energy_filter[i], E_AEV - _energy_filter[i]))
        assert(abs(E_AEV-_energy_filter[i]) < 1e-14)

@pytest.mark.higheffort
@pytest.mark.parametrize(
    "n, basics_options",
    [
        #Here the "subsystem" is already the entire system
        #These are included here as less convoluted tests
        #Having them pass but the real subsystem fail helps 
        (
            [1, 2],
            {"subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)]]}
        ),
        #Mix qubit order to get non-standard one
        (
            [2, 2],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,1),
                                    cirq.GridQubit(1,0), cirq.GridQubit(0,1)]]}
        ),
        #Test [2,4] block as largest block in real sim
        (
            [2, 4],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(0,1), cirq.GridQubit(1,1),
                                    cirq.GridQubit(0,2), cirq.GridQubit(1,2), cirq.GridQubit(0,3), cirq.GridQubit(1,3)]]}
        ),
        #Below the "subsystem" is already the entire system
        # 2 subsystems
        (
            [2, 2],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
                                    [cirq.GridQubit(1,0), cirq.GridQubit(1,1)]]}
        ),
        (
            [2, 2],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
                                    [cirq.GridQubit(0,1), cirq.GridQubit(1,1)]]}
        ),
        # 3 and more subsystems
        (
            [2, 3],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
                                    [cirq.GridQubit(0,1), cirq.GridQubit(1,1)],
                                    [cirq.GridQubit(0,2), cirq.GridQubit(1,2)]]}
        ),
        (
            [3, 2],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
                                    [cirq.GridQubit(1,0), cirq.GridQubit(1,1)],
                                    [cirq.GridQubit(2,0), cirq.GridQubit(2,1)]]}
        ),
        #Mix up subsystem qubit order
        #Mixing subsystems work
        (
            [3, 2],
            {"subsystem_qubits": [  [cirq.GridQubit(1,0), cirq.GridQubit(1,1)],
                                    [cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
                                    [cirq.GridQubit(2,0), cirq.GridQubit(2,1)]]}
        ),
        #Mixing qubit order within the subsystems still fails:
        #(
        #    [3, 2],
        #    {"subsystem_qubits": [  [cirq.GridQubit(1,0), cirq.GridQubit(1,1)],
        #                            [cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
        #                            [cirq.GridQubit(2,1), cirq.GridQubit(2,0)]]}
        #),
        (
            [2, 4],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
                                    [cirq.GridQubit(0,1), cirq.GridQubit(0,2), cirq.GridQubit(1,1), cirq.GridQubit(1,2)],
                                    [cirq.GridQubit(0,3), cirq.GridQubit(1,3)]]}
        ),
        (
            [4, 2],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
                                    [cirq.GridQubit(1,0), cirq.GridQubit(1,1), cirq.GridQubit(2,0), cirq.GridQubit(2,1)],
                                    [cirq.GridQubit(3,0), cirq.GridQubit(3,1)]]}
        ),
        (
            [3, 3],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(2,0)],
                                    [cirq.GridQubit(0,1), cirq.GridQubit(1,1), cirq.GridQubit(2,1)],
                                    [cirq.GridQubit(0,2), cirq.GridQubit(1,2), cirq.GridQubit(2,2)]]}
        ),
        # This takes long and fails
        #(
        #    [3, 4],
        #    {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(2,0)],
        #                            [cirq.GridQubit(0,1), cirq.GridQubit(1,1), cirq.GridQubit(2,1),
        #                            cirq.GridQubit(0,2), cirq.GridQubit(1,2), cirq.GridQubit(2,2)],
        #                            [cirq.GridQubit(0,3), cirq.GridQubit(1,3), cirq.GridQubit(2,3)]]}
        #),
        #This works but takes long
        (
            [3, 4],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(2,0)],
                                    [cirq.GridQubit(0,1), cirq.GridQubit(1,1), cirq.GridQubit(2,1)],
                                    [cirq.GridQubit(0,2), cirq.GridQubit(1,2), cirq.GridQubit(2,2)],
                                    [cirq.GridQubit(0,3), cirq.GridQubit(1,3), cirq.GridQubit(2,3)]]}
        ),
        #Test coverings like used in later simulations:
        #1x4, 4x1 covering
        (
            [4, 4],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(2,0), cirq.GridQubit(3,0)],
                                    [cirq.GridQubit(0,1), cirq.GridQubit(1,1), cirq.GridQubit(2,1), cirq.GridQubit(3,1)],
                                    [cirq.GridQubit(0,2), cirq.GridQubit(1,2), cirq.GridQubit(2,2), cirq.GridQubit(3,2)],
                                    [cirq.GridQubit(0,3), cirq.GridQubit(1,3), cirq.GridQubit(2,3), cirq.GridQubit(3,3)]]}
        ),
        (
            [4, 4],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(0,2), cirq.GridQubit(0,3)],
                                    [cirq.GridQubit(1,0), cirq.GridQubit(1,1), cirq.GridQubit(1,2), cirq.GridQubit(1,3)],
                                    [cirq.GridQubit(2,0), cirq.GridQubit(2,1), cirq.GridQubit(2,2), cirq.GridQubit(2,3)],
                                    [cirq.GridQubit(3,0), cirq.GridQubit(3,1), cirq.GridQubit(3,2), cirq.GridQubit(3,3)]]}
        ),
        #2x4, 1x4_2x4_1x4 covering
        (
            [4, 4],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(0,2), cirq.GridQubit(0,3),
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1), cirq.GridQubit(1,2), cirq.GridQubit(1,3)],
                                    [cirq.GridQubit(2,0), cirq.GridQubit(2,1), cirq.GridQubit(2,2), cirq.GridQubit(2,3),
                                    cirq.GridQubit(3,0), cirq.GridQubit(3,1), cirq.GridQubit(3,2), cirq.GridQubit(3,3)]]}
        ),
        (
            [4, 4],
            {"subsystem_qubits": [  [cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(0,2), cirq.GridQubit(0,3)],
                                    [cirq.GridQubit(1,0), cirq.GridQubit(1,1), cirq.GridQubit(1,2), cirq.GridQubit(1,3),
                                    cirq.GridQubit(2,0), cirq.GridQubit(2,1), cirq.GridQubit(2,2), cirq.GridQubit(2,3)],
                                    [cirq.GridQubit(3,0), cirq.GridQubit(3,1), cirq.GridQubit(3,2), cirq.GridQubit(3,3)]]}
        ),
    ]
)
def test_subsystem_U2(n, basics_options):
    # Do the same as in test_subsystem_U
    # But take superpositions of eigenstates
    j_v = 2*0.9*(np.random.rand(n[0]-1,n[1])- 0.5)
    j_v = j_v + np.sign(j_v)+0.1
    j_h = 2*0.9*(np.random.rand(n[0],n[1]-1)- 0.5)
    j_v = j_v + np.sign(j_v)+0.1
    h = 0.9*(np.random.rand(n[0],n[1])+ 0.1)

    #This does not actually matter as Ising itself is not used
    ising= Ising("GridQubit", n, j_v, j_h, h, "X")
    ising.set_simulator("cirq", {"dtype": np.complex128})

    _pauligates = [cirq.X, cirq.Y, cirq.Z]
    tmp_SQG = _pauligates[np.random.randint(3)]
    tmp_TQG = lambda q1, q2: _pauligates[np.random.randint(3)](q1)*_pauligates[np.random.randint(3)](q2)

    common_options = {    "start": "exact", 
                          "append": False,
                          "cc_exact": True,
                          "SingleQubitGates": [[tmp_SQG]],
                          "TwoQubitGates": [[tmp_TQG]]}
    common_options.update(basics_options)
    ising.set_circuit("basics", common_options)
    _qubit_order = ising.basics.get_subsystem_qubit_map(ising)


    #Prepare energy comparison
    AExpValue_obj = AbstractExpectationValue(ising,
                                            sum(ising.subsystem_hamiltonians))
    _energy_filter = ising.basics.get_energy_filter_from_subsystem(ising)
    
    #Starting at the nth Ising eigenvector and applying U^-1 = U^dagger
    #Should end up in the nth Z-Eigenstate
    wf0 = np.zeros(2**(n[0]*n[1]))

    #For less than 8 qubits check every basis vector
    #Otherwise check a random sample
    if n[0]*n[1] < 9:
        indices = range(2**(n[0]*n[1]))
    else:
        rng = np.random.default_rng()
        indices=rng.integers(low=0, high=2**(n[0]*n[1]), size=16)

    previous_i1 = 0
    previous_i2 = 0
    for iteration in range(int(len(indices)/2)):
        #choose random mixing ratio
        mix_ratio=np.random.random_sample()
        
        i1 = indices[2*iteration]
        # Choose index far away and odd
        # Idea: large energy difference
        i2 = indices[np.mod(    2*iteration+1+int(len(indices)/2), 
                                len(indices))]

        #Get composite subststem eigenstate by tensorproduct
        if len(ising.subsystem_qubits) == 1:
            in_state = (np.sqrt(mix_ratio)*ising.eig_vec[:,i1])+(np.sqrt(1-mix_ratio)*ising.eig_vec[:,i2])            
        else:
            # Maybe this is a function that also should be provided in circuits.basics
            # i to binary
            # cut subsystems accordingly
            # retransform to int to use in ising.subsystem_U[0][:,3] etc

            in_states = []
            for i in [i1, i2]:
                tmp_binary = np.binary_repr(i, width=n[0]*n[1])

                i_sub_0 = int(tmp_binary[:len(ising.subsystem_qubits[0])],2)
                tmp_binary = tmp_binary[len(ising.subsystem_qubits[0]):]
                #print(i_sub_0, tmp_binary)

                i_sub_1 = int(tmp_binary[:len(ising.subsystem_qubits[1])],2)
                tmp_binary = tmp_binary[len(ising.subsystem_qubits[1]):]
                #print(i_sub_1, tmp_binary)

                in_state= np.tensordot( ising.subsystem_U[0][:,i_sub_0], 
                                        ising.subsystem_U[1][:,i_sub_1], 
                                        axes=0).reshape(2**(len(ising.subsystem_qubits[0])+len(ising.subsystem_qubits[1])))

                #To do add code for len(ising.subsystem_qubits) > 2
                for n_sub in range(2, len(ising.subsystem_qubits)):
                    i_sub = int(tmp_binary[:len(ising.subsystem_qubits[n_sub])],2)
                    tmp_binary = tmp_binary[len(ising.subsystem_qubits[n_sub]):]

                    in_state = np.tensordot( in_state, 
                                            ising.subsystem_U[n_sub][:,i_sub], 
                                            axes=0).reshape(len(in_state)*2**(len(ising.subsystem_qubits[n_sub])))

                in_states.append(in_state)
            in_state= (np.sqrt(mix_ratio)*in_states[0])+(np.sqrt(1-mix_ratio)*in_states[1])
            
        
        #This tests U^\dagger |\phi_m^(HA)> \otimes |\phi_l^(HB)> = |m> \otimes |l>
        wf = ising.simulator.simulate(  initial_state=in_state,
                                        program=ising.circuit,
                                        qubit_order=_qubit_order).state_vector()
        wf = wf/np.linalg.norm(wf)

        wf0[previous_i1]=0
        wf0[previous_i2]=0

        previous_i1 = i1
        previous_i2 = i2

        wf0[i1]=np.sqrt(mix_ratio)
        wf0[i2]=np.sqrt(1-mix_ratio)
        
        print("i1: {}\ti2: {}\nwf: {}\nwf0: {}".format(i1, i2,wf, wf0))
        cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(wf, wf0, rtol=n[0]*n[1]*1e-14, atol=n[0]*n[1]*5e-14)
        
        #use here qubit order for subsystem = system
        E_filter= mix_ratio*_energy_filter[i1]+(1-mix_ratio)*_energy_filter[i2]

        if len(ising.subsystem_qubits) == 1:
            E_AEV = AExpValue_obj.evaluate( atol=1e-14,
                                            wavefunction=in_state)/(n[0]*n[1])
        else:
            E_AEV = AExpValue_obj.evaluate( atol=1e-14,
                                            q_map=_qubit_order,
                                            wavefunction=in_state)/(n[0]*n[1])
        print("Energy from AbstractExpectationValue: \t{}\nEnergy from energy filter: \t\t{}\nDifference: \t\t\t\t {}"
            .format(E_AEV,E_filter, E_AEV - E_filter))
       
        assert(abs(E_AEV-E_filter) < 1e-14)

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
    #print("j_v: {}\nj_h {}\nh {}".format(j_v, j_h, h))
    
    ising= Ising("GridQubit", n, j_v, j_h, h, "X")
    ising.set_circuit("basics",{    "start": "exact", 
                                    "n_exact": n_exact})

    #print("ising.circuit:\n{}\n".format(ising.circuit))

    ising2= Ising("GridQubit", n, j_v, j_h, h, "X")
    ising2.set_circuit("basics",{   "start": "exact", 
                                    "subsystem_qubits": subsystem_qubits})
    #print("ising2.circuit:\n{}\n".format(ising2.circuit))

    #Only compare unitaries if circuits do not correspond
    if ising.circuit == ising2.circuit:
        assert True
    else:
        print("ising.circuit.all_qubits():\n{}\nising2.circuit.all_qubits():\n{}\n".format(ising.circuit.all_qubits(),ising2.circuit.all_qubits()))
        print("ising.circuit.unitary()-ising2.circuit.unitary()\n{}\n".format(ising.circuit.unitary()-ising2.circuit.unitary()))
        cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(ising.circuit.unitary(),ising2.circuit.unitary(), rtol=1e-7, atol=1e-7)

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

    #print("ising.circuit:\n{}\n".format(ising.circuit))
    h0 = 2*(np.random.rand(n[0],n[1])- 0.5)
    ising2 = Ising("GridQubit", n, j_v, j_h, h0, "X")
    ising2.set_circuit("basics",{    "start": "exact", 
                                    "subsystem_qubits": subsystem_qubits,
                                    "subsystem_h": subsystem_h})
    #print("ising2.circuit:\n{}\n".format(ising2.circuit))

    #Only compare unitaries if circuits do not correspond
    if ising.circuit == ising2.circuit:
        assert True
    else:
        print("ising.circuit.all_qubits():\n{}\nising2.circuit.all_qubits():\n{}\n".format(ising.circuit.all_qubits(),ising2.circuit.all_qubits()))
        print("ising.circuit.unitary()-ising2.circuit.unitary()\n{}\n".format(ising.circuit.unitary()-ising2.circuit.unitary()))
        cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(ising.circuit.unitary(),ising2.circuit.unitary(), rtol=1e-12, atol=1e-12)

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
    #print("j_v: {}\nj_h {}\nh {}".format(j_v, j_h, h))
    ising= Ising("GridQubit", n, j_v, j_h, h, "X")
    ising.set_circuit("basics",{   "start": "exact", 
                                    "n_exact": n_exact})

    #print("ising.circuit:\n{}\n".format(ising.circuit))

    #print("subsystem_j_v:\n{}\nsubsystem_j_h:\n{}".format(subsystem_j_v, subsystem_j_h))
    j_v0 = 2*(np.random.rand(n[0]-1,n[1])- 0.5)
    j_h0 = 2*(np.random.rand(n[0],n[1]-1)- 0.5)
    ising2 = Ising("GridQubit", n, j_v0, j_h0, h, "X")
    ising2.set_circuit("basics",{    "start": "exact", 
                                    "subsystem_qubits": subsystem_qubits,
                                    "subsystem_j_h": subsystem_j_h,
                                    "subsystem_j_v": subsystem_j_v})
    #print("ising2.circuit:\n{}\n".format(ising2.circuit))

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

@pytest.mark.parametrize(
    "n,basics_options,SingleQubitGates, TwoQubitGates, subsystem_hamiltonians",
    [
        (
            [1, 2],
            { "n_exact": [1, 2]},
            [[cirq.X]],
            [[lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)]],
            [-cirq.Z(cirq.GridQubit(0,0))*cirq.Z(cirq.GridQubit(0,1))-cirq.X(cirq.GridQubit(0,0))-cirq.X(cirq.GridQubit(0,1))],
        ),
        (
            [2, 2],
            { "n_exact": [1, 2]},
            [[cirq.Y]],
            [[lambda q1, q2: cirq.X(q1)*cirq.Z(q2)]],
            [-cirq.X(cirq.GridQubit(0,0))*cirq.Z(cirq.GridQubit(0,1))-cirq.Y(cirq.GridQubit(0,0))-cirq.Y(cirq.GridQubit(0,1)),
            -cirq.X(cirq.GridQubit(1,0))*cirq.Z(cirq.GridQubit(1,1))-cirq.Y(cirq.GridQubit(1,0))-cirq.Y(cirq.GridQubit(1,1))],
        ),
        (
            [2, 2],
            { "subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
                                    [cirq.GridQubit(1,0), cirq.GridQubit(1,1)]]},
            [[cirq.Y]],
            [[lambda q1, q2: cirq.X(q1)*cirq.Z(q2)]],
            [-cirq.X(cirq.GridQubit(0,0))*cirq.Z(cirq.GridQubit(0,1))-cirq.Y(cirq.GridQubit(0,0))-cirq.Y(cirq.GridQubit(0,1)),
            -cirq.X(cirq.GridQubit(1,0))*cirq.Z(cirq.GridQubit(1,1))-cirq.Y(cirq.GridQubit(1,0))-cirq.Y(cirq.GridQubit(1,1))],
        ),
        #If it works to have qubits = subsystem_qubits
        #but some subsystem qubits excluded by j_v, j_h, h =0
        (
            [2, 2],
            { "subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(1,0), cirq.GridQubit(1,1)], 
                                    [cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
                "subsystem_j_v":[   np.transpose(np.array([ [[0]], [[0]]]), (1,0, 2)),
                                    np.transpose(np.array([ [[0]], [[0]]]), (1,0, 2))],
                "subsystem_j_h": [   0.1*np.transpose(np.array([ [[1], [0]]]), (1,0, 2)),
                                    0.2*np.transpose(np.array([ [[0], [1]]]), (1,0, 2))],
                "subsystem_h": [    0.3*np.transpose(np.array([ [[1], [0]], [[1], [0]]]), (1,0, 2)), 
                                    0.4*np.transpose(np.array([ [[0], [1]], [[0], [1]] ]), (1,0, 2)) ]},
            [[cirq.Y]],
            [[lambda q1, q2: cirq.X(q1)*cirq.Z(q2)]],
            [-0.1*cirq.X(cirq.GridQubit(0,0))*cirq.Z(cirq.GridQubit(0,1))-0.3*cirq.Y(cirq.GridQubit(0,0))-0.3*cirq.Y(cirq.GridQubit(0,1)),
            -0.2*cirq.X(cirq.GridQubit(1,0))*cirq.Z(cirq.GridQubit(1,1))-0.4*cirq.Y(cirq.GridQubit(1,0))-0.4*cirq.Y(cirq.GridQubit(1,1))],
        ),
        (
            [2, 2],
            { "subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)], 
                                    [cirq.GridQubit(1,0), cirq.GridQubit(1,1)]]},
            [[cirq.Z]],
            [[lambda q1, q2: cirq.X(q1)*cirq.Z(q2)]],
            [-cirq.X(cirq.GridQubit(0,0))*cirq.Z(cirq.GridQubit(0,1))-cirq.Z(cirq.GridQubit(0,0))-cirq.Z(cirq.GridQubit(0,1)),
            -cirq.X(cirq.GridQubit(1,0))*cirq.Z(cirq.GridQubit(1,1))-cirq.Z(cirq.GridQubit(1,0))-cirq.Z(cirq.GridQubit(1,1))],
        ),
        (
            [2, 2],
            { "subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(1,0)], 
                                    [cirq.GridQubit(0,1), cirq.GridQubit(1,1)]]},
            [[cirq.Z]],
            [[lambda q1, q2: cirq.X(q1)*cirq.Z(q2)]],
            [-cirq.X(cirq.GridQubit(0,0))*cirq.Z(cirq.GridQubit(1,0))-cirq.Z(cirq.GridQubit(0,0))-cirq.Z(cirq.GridQubit(1,0)),
            -cirq.X(cirq.GridQubit(0,1))*cirq.Z(cirq.GridQubit(1,1))-cirq.Z(cirq.GridQubit(0,1))-cirq.Z(cirq.GridQubit(1,1))],
        ),
    ]
)
def test__exact_layer_subsystem_hamiltonians(n,basics_options,SingleQubitGates, TwoQubitGates, subsystem_hamiltonians):
    j_v0 = np.ones((n[0]-1,n[1]))
    j_h0 = np.ones((n[0],n[1]-1))
    h0 = np.ones((n[0],n[1]))

    spinmodel = SpinModel("GridQubit", 
                            n, 
                            [j_v0], 
                            [j_h0], 
                            [h0],
                            [lambda q1, q2: cirq.Y(q1)*cirq.Y(q2)],
                            [cirq.Y])
    options = {    "start": "exact",
                    "b_exact": [1,1],
                    "SingleQubitGates": SingleQubitGates,
                    "TwoQubitGates": TwoQubitGates,
                    "subsystem_diagonalisation": False}
    options.update(basics_options)
    spinmodel.set_circuit("basics",options)

    for i in range(len(subsystem_hamiltonians)):
        print("spinmodel.subsystem_hamiltonians[{}]\n{}\nsubsystem_hamiltonians[{}]\n{}\n".format(i, spinmodel.subsystem_hamiltonians[i],i, subsystem_hamiltonians[i]))
    assert(all([spinmodel.subsystem_hamiltonians[i]==subsystem_hamiltonians[i] for i in range(len(subsystem_hamiltonians))]))

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
            [2,3]
        ),
        (
            [3,2]
        ),
    ]
)
def test_get_energy_filter_from_subsystem1(n):
    j_v0 = 2*(np.random.rand(n[0]-1,n[1])- 0.5)
    j_h0 = 2*(np.random.rand(n[0],n[1]-1)- 0.5)
    h0 = 2*(np.random.rand(n[0],n[1])- 0.5)
    ising = Ising("GridQubit", n, j_v0, j_h0, h0, "X")

    converter_obj = Converter()
    scipy_sparse_matrix=converter_obj.cirq_paulisum2scipy_crsmatrix(ising.hamiltonian, dtype=np.float64)
    ising.diagonalise(  solver = "scipy", 
                        solver_options={"subset_by_index": [0, 2**(n[0]*n[1]) - 1]},
                        matrix=scipy_sparse_matrix.toarray())

    #get energy filter
    energy_filter = ising.basics.get_energy_filter_from_subsystem(ising, ising.eig_val)
    #change a value in ising.eig_val randomly to confirm it is indeed a view()/pointer link
    ising.eig_val[0] = np.random.random_sample()
    assert((ising.eig_val == energy_filter).all())

@pytest.mark.parametrize(
    "n, HA_options, HB_options",
    [
        (
            [1,2],
            {"subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)]]},
            {   "subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)]],
                "subsystem_h" : [ np.transpose(np.array([[[0], [0]]]), (0, 1,2))],
                "subsystem_j_h" :[    np.transpose(np.array([ [[0]]]), (1,0, 2)) ]  }
        ),
        (
            [1,2],
            {   "subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)]],
                "subsystem_j_h" :[    np.transpose(np.array([ [[0]]]), (1,0, 2))] },
            {   "subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)]],
                "subsystem_h" : [ np.transpose(np.array([[[0], [0]]]), (0, 1,2))]}
        ),
        (
            [2,2],
            {"subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]]},
            {   "subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
                "subsystem_h" : [ np.transpose(np.array([[[0], [0]], [[0], [0]]]), (0, 1,2))],
                "subsystem_j_h" :[    np.transpose(np.array([ [[0], [0]]]), (1,0, 2)) ] ,
                "subsystem_j_v" :[    np.transpose(np.array([ [[0]], [[0]]]), (1,0, 2)) ] ,   }
        ),
        (
            [2,2],
            {"subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
            "subsystem_h" : [ np.transpose(np.array([[[0], [0]], [[0], [0]]]), (0, 1,2))],},
            {   "subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
                "subsystem_j_h" :[    np.transpose(np.array([ [[0], [0]]]), (1,0, 2)) ] ,
                "subsystem_j_v" :[    np.transpose(np.array([ [[0]], [[0]]]), (1,0, 2)) ] ,   }
        ),
                (
            [2,2],
            {"subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
            "subsystem_j_v" :[    np.transpose(np.array([ [[0]], [[0]]]), (1,0, 2)) ]},
            {   "subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
                "subsystem_h" : [ np.transpose(np.array([[[0], [0]], [[0], [0]]]), (0, 1,2))],
                "subsystem_j_h" :[    np.transpose(np.array([ [[0], [0]]]), (1,0, 2)) ]  }
        ),
    ]
)
def test_get_energy_filter_from_subsystem2(n,HA_options,HB_options):
    j_v0 = 2*(np.random.rand(n[0]-1,n[1])- 0.5)
    j_h0 = 2*(np.random.rand(n[0],n[1]-1)- 0.5)
    h0 = 2*(np.random.rand(n[0],n[1])- 0.5)
    ising = Ising("GridQubit", n, j_v0, j_h0, h0, "X")

    # Calculate ground state and first excited state by exact diagonalisation
    converter_obj = Converter()
    scipy_sparse_matrix=converter_obj.cirq_paulisum2scipy_crsmatrix(ising.hamiltonian, dtype=np.float64)
    ising.diagonalise( matrix=scipy_sparse_matrix)
    print("Hamiltonian:\n{}\n".format(ising.hamiltonian))
    print("Exact energies: {}".format(ising.eig_val))
    E0 = ising.eig_val[0],
    phi0 = np.copy(ising.eig_vec[:,0].astype(np.complex64))
    #print("Phi0: {}".format(phi0))

    #For HA and HB Common basics_options
    common_basics_options={"start": "exact", 
                        "append": False, 
                        "subsystem_diagonalisation": True,
                        "b_exact": [0,0],
                        "cc_exact": True}

    # H_A
    # 1. Set rotation circuit 
    # 2. Get energy filter
    # 3. Calculate <\phi|H_A|phi> in H_A eigen basis
    basics_options = common_basics_options.copy()
    basics_options.update(HA_options)
    ising.set_circuit("basics",basics_options)
    print("H_A:\n{}\n".format(ising.subsystem_hamiltonians[0]))
    #print("H_A rotation circuit:\n{}\n".format(ising.circuit))
    #print("Phi0: {}".format(phi0))
    
    energy_filter_A = ising.basics.get_energy_filter_from_subsystem(ising)
    wf_HA_basis = ising.simulator.simulate( ising.circuit, 
                                            initial_state = phi0).state_vector()
    E_A = np.vdot(energy_filter_A, abs(wf_HA_basis)**2)

    # H_B
    # 1. Set rotation circuit 
    # 2. Get energy filter
    # 3. Calculate <\phi|H_B|phi> in H_B eigen basis
    basics_options = common_basics_options.copy()
    basics_options.update(HB_options)
    #print(basics_options)
    ising.set_circuit("basics",basics_options)
    print("H_B:\n{}\n".format(ising.subsystem_hamiltonians[0]))
    #print("H_B rotation circuit:\n{}\n".format(ising.circuit))
    #print("Phi0: {}".format(phi0))
    
    energy_filter_B = ising.basics.get_energy_filter_from_subsystem(ising)
    wf_HB_basis = ising.simulator.simulate( ising.circuit, 
                                            initial_state = phi0).state_vector()
    E_B = np.vdot(energy_filter_B, abs(wf_HB_basis)**2)
    
    #Assert ising.eig_val = <\phi|H_A|phi> + <\phi|H_B|phi>
    print("E0: {}\tEA: {}\tEB: {}".format(E0,E_A, E_B))
    assert(abs(E0 -(E_A+E_B)) < 1e-7)

@pytest.mark.parametrize(
    "n, HA_options, HB_options",
    [
        (
            [1,2],
            {"subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)]]},
            {   "subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)]],
                "subsystem_h" : [ np.transpose(np.array([[[0], [0]]]), (0, 1,2))],
                "subsystem_j_h" :[    np.transpose(np.array([ [[0]]]), (1,0, 2)) ]  }
        ),
        (
            [1,2],
            {   "subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)]],
                "subsystem_j_h" :[    np.transpose(np.array([ [[0]]]), (1,0, 2))] },
            {   "subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1)]],
                "subsystem_h" : [ np.transpose(np.array([[[0], [0]]]), (0, 1,2))]}
        ),
        (
            [2,2],
            {"subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]]},
            {   "subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
                "subsystem_h" : [ np.transpose(np.array([[[0], [0]], [[0], [0]]]), (0, 1,2))],
                "subsystem_j_h" :[    np.transpose(np.array([ [[0], [0]]]), (1,0, 2)) ] ,
                "subsystem_j_v" :[    np.transpose(np.array([ [[0]], [[0]]]), (1,0, 2)) ] ,   }
        ),
        (
            [2,2],
            {"subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
            "subsystem_h" : [ np.transpose(np.array([[[0], [0]], [[0], [0]]]), (0, 1,2))],},
            {   "subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
                "subsystem_j_h" :[    np.transpose(np.array([ [[0], [0]]]), (1,0, 2)) ] ,
                "subsystem_j_v" :[    np.transpose(np.array([ [[0]], [[0]]]), (1,0, 2)) ] ,   }
        ),
                (
            [2,2],
            {"subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
            "subsystem_j_v" :[    np.transpose(np.array([ [[0]], [[0]]]), (1,0, 2)) ]},
            {   "subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
                "subsystem_h" : [ np.transpose(np.array([[[0], [0]], [[0], [0]]]), (0, 1,2))],
                "subsystem_j_h" :[    np.transpose(np.array([ [[0], [0]]]), (1,0, 2)) ]  }
        ),
    ]
)
def test_get_energy_filter_from_subsystem3(n,HA_options,HB_options):
    j_v0 = 2*(np.random.rand(n[0]-1,n[1])- 0.5)
    j_h0 = 2*(np.random.rand(n[0],n[1]-1)- 0.5)
    h0 = 2*(np.random.rand(n[0],n[1])- 0.5)
    ising = Ising("GridQubit", n, j_v0, j_h0, h0, "X")
    ising.set_simulator("cirq")

    # Calculate energy of random state
    expval_obj = ExpectationValue(ising)
    state=np.random.rand(1,2**(n[0]*n[1])) + 1j*np.random.rand(1,2**(n[0]*n[1])) 
    state=np.squeeze(state)/np.linalg.norm(state)
    E = expval_obj.evaluate(state)

    #For HA and HB Common basics_options
    common_basics_options={"start": "exact", 
                        "append": False, 
                        "subsystem_diagonalisation": True,
                        "b_exact": [0,0],
                        "cc_exact": True}

    # H_A
    # 1. Set rotation circuit 
    # 2. Get energy filter
    # 3. Calculate <\phi|H_A|phi> in H_A eigen basis
    basics_options = common_basics_options.copy()
    basics_options.update(HA_options)
    ising.set_circuit("basics",basics_options)
    hamiltonian_HA = sum(ising.subsystem_hamiltonians)
    print("H_A:\n{}\n".format(hamiltonian_HA))
    #print("H_A rotation circuit:\n{}\n".format(ising.circuit))
    #print("Phi0: {}".format(phi0))
    
    energy_filter_A = ising.basics.get_energy_filter_from_subsystem(ising)
    wf_HA_basis = ising.simulator.simulate( ising.circuit, 
                                            initial_state = state).state_vector()
    E_A = np.vdot(energy_filter_A, abs(wf_HA_basis)**2)

    # H_B
    # 1. Set rotation circuit 
    # 2. Get energy filter
    # 3. Calculate <\phi|H_B|phi> in H_B eigen basis
    basics_options = common_basics_options.copy()
    basics_options.update(HB_options)
    #print(basics_options)
    ising.set_circuit("basics",basics_options)
    hamiltonian_HB = sum(ising.subsystem_hamiltonians)
    print("H_B:\n{}\n".format(hamiltonian_HB))
    #print("H_B rotation circuit:\n{}\n".format(ising.circuit))
    #print("Phi0: {}".format(phi0))
    
    energy_filter_B = ising.basics.get_energy_filter_from_subsystem(ising)
    wf_HB_basis = ising.simulator.simulate( ising.circuit, 
                                            initial_state = state).state_vector()
    E_B = np.vdot(energy_filter_B, abs(wf_HB_basis)**2)
    
    #Assert H = H_A + H_B
    assert(ising.hamiltonian == (hamiltonian_HA+hamiltonian_HB))
    #Assert ising.eig_val = <\phi|H_A|phi> + <\phi|H_B|phi>
    print("E: {}\tEA: {}\tEB: {}".format(E,E_A, E_B))
    assert(abs(E -(E_A+E_B)) < 1e-7)

@pytest.mark.parametrize(
    "n, HA_options, HB_options",
    [
        #Note here all qubits are in the subsystem 
        #Even so the layout is
        #    +  +        x--x
        #    |  |    &   
        #    +  +        x--x
         (
            [2,2],
            {"subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
            "subsystem_h" :     [   0*0.5*np.transpose(np.array([[[1], [1]], [[1], [1]]]), (0, 1,2))],
            "subsystem_j_v" :   [   np.transpose(np.array([ [[0]], [[0]]]), (1,0, 2)) ]},
            {   "subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
                "subsystem_h" :     [   2*0.5*np.transpose(np.array([[[1], [1]], [[1], [1]]]), (0, 1,2))],
                "subsystem_j_h" :   [    np.transpose(np.array([ [[0], [0]]]), (1,0, 2)) ]  }
        ),
        (
            [2,2],
            {"subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
            "subsystem_j_v" :[    np.transpose(np.array([ [[0]], [[0]]]), (1,0, 2)) ]},
            {   "subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
                "subsystem_h" : [ np.transpose(np.array([[[0], [0]], [[0], [0]]]), (0, 1,2))],
                "subsystem_j_h" :[    np.transpose(np.array([ [[0], [0]]]), (1,0, 2)) ]  }
        ),
        #Here the layout is also
        #    +  +        x--x
        #    |  |    &   
        #    +  +        x--x
        #But actually only diagonalise the subsystems
        #Compare ising.circuit
        #(
        #    [2,2],
        #    {   "subsystem_qubits": [   [   cirq.GridQubit(0,0), cirq.GridQubit(0,1)], 
        #                                [   cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
        #        "subsystem_h" :     [   0.5*np.transpose(np.array([[[1], [1]]]), (0, 1,2)),
        #                                0.5*np.transpose(np.array([[[1], [1]]]), (0, 1,2))]},
        #        #"subsystem_j_v" :   [   np.transpose(np.array([ [[0]]]), (1,0, 2)),
        #        #                        np.transpose(np.array([ [[0]]]), (1,0, 2)) ]},
        #    {   "subsystem_qubits": [   [   cirq.GridQubit(0,0), cirq.GridQubit(1,0)], 
        #                                [   cirq.GridQubit(0,1), cirq.GridQubit(1,1)]],
        #        "subsystem_h" :     [   0.5*np.transpose(np.array([[[1]], [[1]]]), (0, 1,2)),
        #                                0.5*np.transpose(np.array([[[1]], [[1]]]), (0, 1,2))]},
        #        #"subsystem_j_h" :   [   np.transpose(np.array([ [[0]]]), (1,0, 2)),
        #        #                        np.transpose(np.array([ [[0]]]), (1,0, 2)),]  }
        #),
    ]
)
def test_get_energy_filter_from_subsystem4(n,HA_options,HB_options):
    """
    This test:
        Use random state vector
        Use random J and h

        Compare different H_A and H_B partitions, e.g.:
                    x--x--x--x       
                    |  |  |  |      
                    x--x--x--x      

                        vs.

            +--+--+--+         x  x  x  x
            |  |  |  |     &   
            +--+--+--+         x  x  x  x
            
                        vs.

            x  x--x  x         x--x  x--x
            |  |  |  |     &   |  |  |  |
            x  x--x  x         x--x  x--x
    """
    #h= np.random.rand(1,1)
    #J= 2*(np.random.rand(1,1)- 0.5)
    h = 0.7; J = 1;
    #j_v0 = J*np.ones((n[0]-1,n[1]))
    #j_h0 = 0.3*J*np.ones((n[0],n[1]-1))
    h0 = h*np.ones((n[0],n[1]))
    j_v0 = J*np.arange(1, 1+ (n[0]-1)*n[1]).reshape((n[0]-1,n[1]))
    j_h0 = 0.3*J*np.arange(1, 1+ (n[1]-1)*n[0]).reshape((n[0],n[1]-1))
    #h0 = h*np.arange(1, 1+ n[1]*n[0]).reshape((n[0],n[1]))
    ising = Ising("GridQubit", n, j_v0, j_h0, h0, "X")
    ising.set_simulator("cirq", {"dtype": np.complex128})
    print("Hamiltonian:\n{}\n".format(ising.hamiltonian))


    # Calculate energy of random state
    expval_obj = ExpectationValue(ising)
    aexpval_obj = AbstractExpectationValue(ising)
    #state=np.random.rand(1,2**(n[0]*n[1])) + 1j*np.random.rand(1,2**(n[0]*n[1])) 
    #state=np.squeeze(state)/np.linalg.norm(state)

    state = np.zeros(2**(n[0]*n[1])).astype(np.complex128)
    #state[0] = 0.5; state[1] = 0.5; state[2] = 0.5; state[3] = 0.5
    state[15] = 1;
    print("np.linalg.norm(state): {}".format(np.linalg.norm(state)))

    E = expval_obj.evaluate(state)
    #E = aexpval_obj.evaluate(state)/(n[0]*n[1])

     #For HA and HB Common basics_options
    common_basics_options={ "start": "exact", 
                            "append": False, 
                            "subsystem_diagonalisation": True,
                            "b_exact": [0,0],
                            "cc_exact": False}

    # H_A
    # 1. Set rotation circuit 
    # 2. Get energy filter
    # 3. Calculate <\phi|H_A|phi> in H_A eigen basis
    basics_options = common_basics_options.copy()
    for key, coefficent in zip(["subsystem_h", "subsystem_j_h", "subsystem_j_v"],[h,J,J]):
        if HA_options.get(key) is not None:
            HA_options[key] = [coefficent*subsystem for subsystem in HA_options.get(key)]

    basics_options.update(HA_options)
    ising.set_circuit("basics",basics_options)
    hamiltonian_HA = sum(ising.subsystem_hamiltonians)
    print("\nH_A:\n{}\n".format(hamiltonian_HA))
    #print("H_A rotation circuit:\n{}\n".format(ising.circuit))
    #print("Phi0: {}".format(phi0))
    
    qubit_map_A = ising.basics.get_subsystem_qubit_map(ising)
    print("qubit_map_A: {}".format(qubit_map_A))
    energy_filter_A = ising.basics.get_energy_filter_from_subsystem(ising)
    wf_HA_basis = ising.simulator.simulate( ising.circuit, 
                                            initial_state = state,
                                            qubit_order=qubit_map_A,
                                            ).state_vector()
    print("energy_filter_A:\n {}".format(energy_filter_A))
    #print(np.round(abs(state)**2, decimals=3))
    print("abs(wf_HA_basis)**2:\n {}".format(np.round(abs(wf_HA_basis)**2, decimals=3)))
    E_A = np.dot(energy_filter_A, abs(wf_HA_basis)**2)
    
    AExpValue_obj = AbstractExpectationValue(ising,
                                            sum(ising.subsystem_hamiltonians)) 
    E_A_AEV = AExpValue_obj.evaluate( atol=1e-14,
                                   #q_map=qubit_map_A,
                                   wavefunction=state)/(n[0]*n[1])

    # H_B
    # 1. Set rotation circuit 
    # 2. Get energy filter
    # 3. Calculate <\phi|H_B|phi> in H_B eigen basis
    basics_options = common_basics_options.copy()
    for key, coefficent in zip(["subsystem_h", "subsystem_j_h", "subsystem_j_v"],[h,J,J]):
        if HB_options.get(key) is not None:
            HB_options[key] = [coefficent*subsystem for subsystem in HB_options.get(key)]

    basics_options.update(HB_options)
    #print(basics_options)
    ising.set_circuit("basics",basics_options)
    hamiltonian_HB = sum(ising.subsystem_hamiltonians)
    print("\nH_B:\n{}\n".format(hamiltonian_HB))
    #print("H_B rotation circuit:\n{}\n".format(ising.circuit))
    #print("Phi0: {}".format(phi0))
    
    qubit_map_B = ising.basics.get_subsystem_qubit_map(ising)
    print("qubit_map_B: {}".format(qubit_map_B))
    energy_filter_B = ising.basics.get_energy_filter_from_subsystem(ising)
    print("energy_filter_B:\n {}".format(energy_filter_B))

    wf_HB_basis = ising.simulator.simulate( ising.circuit, 
                                            initial_state = state,
    #                                        qubit_order=qubit_map_B,
                                            ).state_vector()
    print("abs(wf_HB_basis)**2:\n {}".format(np.round(abs(wf_HB_basis)**2, decimals=3)))
    
    E_B = np.vdot(energy_filter_B, abs(wf_HB_basis)**2)
    
    AExpValue_obj = AbstractExpectationValue(ising,
                                            sum(ising.subsystem_hamiltonians)) 
    E_B_AEV = AExpValue_obj.evaluate( atol=1e-14,
                                   #q_map=qubit_map_B,
                                   wavefunction=state)/(n[0]*n[1])
    #Assert H = H_A + H_B
    print("H - (H_A + H_B):\n{}".format(ising.hamiltonian -(hamiltonian_HA+hamiltonian_HB)))
    assert(ising.hamiltonian == (hamiltonian_HA+hamiltonian_HB))
    #Assert ising.eig_val = <\phi|H_A|phi> + <\phi|H_B|phi>
    print("\nEA: {} \t EA_AEV: {}\nEB: {} \tEB_AEV: {}\nE: \t\t\t{}\nE-(EA+EB): \t\t{}\nE -(E_A_AEV+E_B_AEV): \t{}".format(E_A,E_A_AEV, E_B, E_B_AEV,E, E -(E_A+E_B), E -(E_A_AEV+E_B_AEV)))
    assert(abs(E -(E_A+E_B)) < 1e-7)
    #assert False

@pytest.mark.parametrize(
    "n, HA_options, HB_options",
    [
        (
            [2,2],
            {"subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
                "subsystem_j_v" :[    np.transpose(np.array([ [[0]], [[0]]]), (1,0, 2)) ]},
            {   "subsystem_qubits": [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1), 
                                    cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
                "subsystem_j_h" :[    np.transpose(np.array([ [[0], [0]]]), (1,0, 2)) ]  }
        ),
        (
            [2,2],
            {   "subsystem_qubits": [   [   cirq.GridQubit(0,0), cirq.GridQubit(0,1)], 
                                        [   cirq.GridQubit(1,0), cirq.GridQubit(1,1)]]},
            {   "subsystem_qubits": [   [   cirq.GridQubit(0,0), cirq.GridQubit(1,0)], 
                                        [   cirq.GridQubit(0,1), cirq.GridQubit(1,1)]]}
        ),
    ]
)
def test_get_energy_filter_from_subsystem5(n,HA_options,HB_options):
    j_v0 = 2*(np.random.rand(n[0]-1,n[1])- 0.5)
    j_h0 = 2*(np.random.rand(n[0],n[1]-1)- 0.5)
    h0 = 2*(np.random.rand(n[0],n[1])- 0.5)
    ising = Ising("GridQubit", n, j_v0, j_h0, h0, "X")
    ising.set_simulator("cirq", {"dtype": np.complex128})
    print("Hamiltonian:\n{}\n".format(ising.hamiltonian))

    # Calculate energy of random state
    expval_obj = ExpectationValue(ising)
    state=np.random.rand(1,2**(n[0]*n[1])) + 1j*np.random.rand(1,2**(n[0]*n[1])) 
    state=np.squeeze(state)/np.linalg.norm(state)
    E = expval_obj.evaluate(state)

    #For HA and HB Common basics_options
    common_basics_options={"start": "exact", 
                        "append": False, 
                        "subsystem_diagonalisation": True,
                        "b_exact": [0,0],
                        "cc_exact": True}

    # H_A
    # 1. Set rotation circuit 
    # 2. Get energy filter
    # 3. Calculate <\phi|H_A|phi> in H_A eigen basis
    basics_options = common_basics_options.copy()

    subsystem_h = []
    subsystem_qubits = HA_options.get("subsystem_qubits")
    for i in range(len(subsystem_qubits)):
        subsystem_h.append(0.5*ising.h[min(subsystem_qubits[i])._row: max(subsystem_qubits[i])._row+1,
                min(subsystem_qubits[i])._col: max(subsystem_qubits[i])._col+1, :])
    HA_options.update({"subsystem_h": subsystem_h})

    basics_options.update(HA_options)
    ising.set_circuit("basics",basics_options)
    hamiltonian_HA = sum(ising.subsystem_hamiltonians)
    print("H_A:\n{}\n".format(hamiltonian_HA))
    #print("H_A rotation circuit:\n{}\n".format(ising.circuit))
    #print("Phi0: {}".format(phi0))
    
    qubit_map_A = ising.basics.get_subsystem_qubit_map(ising)
    print("qubit_map_A: {}\n".format(qubit_map_A))
    energy_filter_A = ising.basics.get_energy_filter_from_subsystem(ising)
    wf_HA_basis = ising.simulator.simulate( ising.circuit, 
                                            qubit_order=qubit_map_A,
                                            initial_state = state).state_vector()
    E_A = np.vdot(energy_filter_A, abs(wf_HA_basis)**2)

    AExpValue_obj = AbstractExpectationValue(ising,
                                            sum(ising.subsystem_hamiltonians)) 
    E_A_AEV = AExpValue_obj.evaluate( atol=1e-14,
                                   #q_map=qubit_map_A,
                                   wavefunction=state)/(n[0]*n[1])

    # H_B
    # 1. Set rotation circuit 
    # 2. Get energy filter
    # 3. Calculate <\phi|H_B|phi> in H_B eigen basis
    basics_options = common_basics_options.copy()

    subsystem_h = []
    subsystem_qubits = HB_options.get("subsystem_qubits")
    for i in range(len(subsystem_qubits)):
        subsystem_h.append(0.5*ising.h[min(subsystem_qubits[i])._row: max(subsystem_qubits[i])._row+1,
                min(subsystem_qubits[i])._col: max(subsystem_qubits[i])._col+1, :])
    HB_options.update({"subsystem_h": subsystem_h})
    
    basics_options.update(HB_options)
    #print(basics_options)
    ising.set_circuit("basics",basics_options)
    hamiltonian_HB = sum(ising.subsystem_hamiltonians)
    print("H_B:\n{}\n".format(hamiltonian_HB))
    #print("H_B rotation circuit:\n{}\n".format(ising.circuit))
    #print("Phi0: {}".format(phi0))
    
    qubit_map_B = ising.basics.get_subsystem_qubit_map(ising)
    print("qubit_map_B: {}\n".format(qubit_map_B))
    energy_filter_B = ising.basics.get_energy_filter_from_subsystem(ising)
    wf_HB_basis = ising.simulator.simulate( program=ising.circuit,
                                            qubit_order=qubit_map_B,
                                            initial_state = state).state_vector()
    wf_HB_basis2 = ising.simulator.simulate( program=cirq.Circuit(),
                                            qubit_order=qubit_map_B,
                                            initial_state = wf_HB_basis).state_vector()                                        
    E_B = np.vdot(energy_filter_B, abs(wf_HB_basis)**2)
    E_B2 = np.vdot(energy_filter_B, abs(wf_HB_basis2)**2)
    print("\nnp.vdot(wf_HB_basis, wf_HB_basis2): {}".format(np.vdot(wf_HB_basis, wf_HB_basis2)))
    
    AExpValue_obj = AbstractExpectationValue(ising,
                                            sum(ising.subsystem_hamiltonians)) 
    E_B_AEV = AExpValue_obj.evaluate( atol=1e-14,
                                   #this makes it not workq_map=qubit_map_B,
                                   wavefunction=state)/(n[0]*n[1])

    #Assert H = H_A + H_B
    assert(ising.hamiltonian == (hamiltonian_HA+hamiltonian_HB))

    #Assert ising.eig_val = <\phi|H_A|phi> + <\phi|H_B|phi>
    print("\nEA: {} \t EA_AEV: {}\nEB: {} \tEB_AEV: {} \tEB2: {}\nE: \t\t\t{}\nE-(EA+EB): \t\t{}\nE -(E_A_AEV+E_B_AEV): \t{}"
            .format(E_A,E_A_AEV, E_B, E_B_AEV, E_B2, E, E -(E_A+E_B), E -(E_A_AEV+E_B_AEV)))
    assert(abs(E -(E_A+E_B)) < 1e-7)

@pytest.mark.parametrize(
    "subsystem_qubits, target_qubit_map",
    [
        (
            [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1)]],
            {cirq.GridQubit(0,0):0, cirq.GridQubit(0,1):1},
        ),
        (
            [[ cirq.GridQubit(1,1), cirq.GridQubit(0,1)]],
            {cirq.GridQubit(1,1):0, cirq.GridQubit(0,1):1},
        ),
        (
            [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1)],
            [ cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
            {cirq.GridQubit(0,0):0, cirq.GridQubit(0,1):1, cirq.GridQubit(1,0):2, cirq.GridQubit(1,1):3},
        ),
        (
            [[ cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
            [ cirq.GridQubit(0,1), cirq.GridQubit(1,1)]],
            {cirq.GridQubit(0,0):0, cirq.GridQubit(0,1):2, cirq.GridQubit(1,0):1, cirq.GridQubit(1,1):3},
        ),
        (
            [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2), cirq.GridQubit(0,3)],
            [ cirq.GridQubit(1,0), cirq.GridQubit(1,1),cirq.GridQubit(1,2), cirq.GridQubit(1,3)]],
            {   cirq.GridQubit(0,0):0, cirq.GridQubit(0,1):1, cirq.GridQubit(0,2):2, cirq.GridQubit(0,3):3,
                cirq.GridQubit(1,0):4, cirq.GridQubit(1,1):5, cirq.GridQubit(1,2):6, cirq.GridQubit(1,3):7},
        ),
        (
            [   [   cirq.GridQubit(0,0), cirq.GridQubit(1,0)],
                [   cirq.GridQubit(0,1), cirq.GridQubit(1,1)],
                [   cirq.GridQubit(0,2), cirq.GridQubit(1,2)],
                [   cirq.GridQubit(0,3), cirq.GridQubit(1,3)]],
            {   cirq.GridQubit(0,0):0, cirq.GridQubit(0,1):2, cirq.GridQubit(0,2):4, cirq.GridQubit(0,3):6,
                cirq.GridQubit(1,0):1, cirq.GridQubit(1,1):3, cirq.GridQubit(1,2):5, cirq.GridQubit(1,3):7},
        ),
         (
            [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2), cirq.GridQubit(0,3)],
            [ cirq.GridQubit(1,0), cirq.GridQubit(1,1),cirq.GridQubit(1,2), cirq.GridQubit(1,3)],
            [ cirq.GridQubit(2,0), cirq.GridQubit(2,1),cirq.GridQubit(2,2), cirq.GridQubit(2,3)],
            [ cirq.GridQubit(3,0), cirq.GridQubit(3,1),cirq.GridQubit(3,2), cirq.GridQubit(3,3)]],
            {   cirq.GridQubit(0,0):0, cirq.GridQubit(0,1):1, cirq.GridQubit(0,2):2, cirq.GridQubit(0,3):3,
                cirq.GridQubit(1,0):4, cirq.GridQubit(1,1):5, cirq.GridQubit(1,2):6, cirq.GridQubit(1,3):7,
                cirq.GridQubit(2,0):8, cirq.GridQubit(2,1):9, cirq.GridQubit(2,2):10, cirq.GridQubit(2,3):11,
                cirq.GridQubit(3,0):12, cirq.GridQubit(3,1):13, cirq.GridQubit(3,2):14, cirq.GridQubit(3,3):15},
        ),
        (
            [   [   cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(2,0), cirq.GridQubit(3,0)],
                [   cirq.GridQubit(0,1), cirq.GridQubit(1,1), cirq.GridQubit(2,1), cirq.GridQubit(3,1)],
                [   cirq.GridQubit(0,2), cirq.GridQubit(1,2), cirq.GridQubit(2,2), cirq.GridQubit(3,2)],
                [   cirq.GridQubit(0,3), cirq.GridQubit(1,3), cirq.GridQubit(2,3), cirq.GridQubit(3,3)]],
            {   cirq.GridQubit(0,0):0, cirq.GridQubit(0,1):4, cirq.GridQubit(0,2):8, cirq.GridQubit(0,3):12,
                cirq.GridQubit(1,0):1, cirq.GridQubit(1,1):5, cirq.GridQubit(1,2):9, cirq.GridQubit(1,3):13,
                cirq.GridQubit(2,0):2, cirq.GridQubit(2,1):6, cirq.GridQubit(2,2):10, cirq.GridQubit(2,3):14,
                cirq.GridQubit(3,0):3, cirq.GridQubit(3,1):7, cirq.GridQubit(3,2):11, cirq.GridQubit(3,3):15},
        ),
        (
            [[ cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2), cirq.GridQubit(0,3),
             cirq.GridQubit(1,0), cirq.GridQubit(1,1),cirq.GridQubit(1,2), cirq.GridQubit(1,3)],
            [ cirq.GridQubit(2,0), cirq.GridQubit(2,1),cirq.GridQubit(2,2), cirq.GridQubit(2,3),
             cirq.GridQubit(3,0), cirq.GridQubit(3,1),cirq.GridQubit(3,2), cirq.GridQubit(3,3)]],
            {   cirq.GridQubit(0,0):0, cirq.GridQubit(0,1):1, cirq.GridQubit(0,2):2, cirq.GridQubit(0,3):3,
                cirq.GridQubit(1,0):4, cirq.GridQubit(1,1):5, cirq.GridQubit(1,2):6, cirq.GridQubit(1,3):7,
                cirq.GridQubit(2,0):8, cirq.GridQubit(2,1):9, cirq.GridQubit(2,2):10, cirq.GridQubit(2,3):11,
                cirq.GridQubit(3,0):12, cirq.GridQubit(3,1):13, cirq.GridQubit(3,2):14, cirq.GridQubit(3,3):15},
        ),
        (
            [   [   cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(2,0), cirq.GridQubit(3,0),
                   cirq.GridQubit(0,1), cirq.GridQubit(1,1), cirq.GridQubit(2,1), cirq.GridQubit(3,1)],
                [   cirq.GridQubit(0,2), cirq.GridQubit(1,2), cirq.GridQubit(2,2), cirq.GridQubit(3,2),
                   cirq.GridQubit(0,3), cirq.GridQubit(1,3), cirq.GridQubit(2,3), cirq.GridQubit(3,3)]],
            {   cirq.GridQubit(0,0):0, cirq.GridQubit(0,1):4, cirq.GridQubit(0,2):8, cirq.GridQubit(0,3):12,
                cirq.GridQubit(1,0):1, cirq.GridQubit(1,1):5, cirq.GridQubit(1,2):9, cirq.GridQubit(1,3):13,
                cirq.GridQubit(2,0):2, cirq.GridQubit(2,1):6, cirq.GridQubit(2,2):10, cirq.GridQubit(2,3):14,
                cirq.GridQubit(3,0):3, cirq.GridQubit(3,1):7, cirq.GridQubit(3,2):11, cirq.GridQubit(3,3):15},
        ),
        (
            [   [   cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(1,0), cirq.GridQubit(1,1),
                   cirq.GridQubit(2,0), cirq.GridQubit(2,1), cirq.GridQubit(3,0), cirq.GridQubit(3,1)],
                [   cirq.GridQubit(0,2), cirq.GridQubit(0,3), cirq.GridQubit(1,2), cirq.GridQubit(1,3),
                   cirq.GridQubit(2,2), cirq.GridQubit(2,3), cirq.GridQubit(3,2), cirq.GridQubit(3,3)]],
            {   cirq.GridQubit(0,0):0, cirq.GridQubit(0,1):1, cirq.GridQubit(0,2):8, cirq.GridQubit(0,3):9,
                cirq.GridQubit(1,0):2, cirq.GridQubit(1,1):3, cirq.GridQubit(1,2):10, cirq.GridQubit(1,3):11,
                cirq.GridQubit(2,0):4, cirq.GridQubit(2,1):5, cirq.GridQubit(2,2):12, cirq.GridQubit(2,3):13,
                cirq.GridQubit(3,0):6, cirq.GridQubit(3,1):7, cirq.GridQubit(3,2):14, cirq.GridQubit(3,3):15},
        ),
        (
            [   [   cirq.GridQubit(0,0), cirq.GridQubit(1,0), cirq.GridQubit(2,0), cirq.GridQubit(3,0)],
                [   cirq.GridQubit(0,1), cirq.GridQubit(0,2), cirq.GridQubit(1,1), cirq.GridQubit(1,2),
                   cirq.GridQubit(2,1), cirq.GridQubit(2,2), cirq.GridQubit(3,1), cirq.GridQubit(3,2)],
                [   cirq.GridQubit(0,3), cirq.GridQubit(1,3), cirq.GridQubit(2,3), cirq.GridQubit(3,3)]],
            {   cirq.GridQubit(0,0):0, cirq.GridQubit(0,1):4, cirq.GridQubit(0,2):5, cirq.GridQubit(0,3):12,
                cirq.GridQubit(1,0):1, cirq.GridQubit(1,1):6, cirq.GridQubit(1,2):7, cirq.GridQubit(1,3):13,
                cirq.GridQubit(2,0):2, cirq.GridQubit(2,1):8, cirq.GridQubit(2,2):9, cirq.GridQubit(2,3):14,
                cirq.GridQubit(3,0):3, cirq.GridQubit(3,1):10, cirq.GridQubit(3,2):11, cirq.GridQubit(3,3):15},
        ),
        (
            [   [   cirq.GridQubit(0,0), cirq.GridQubit(0,1), cirq.GridQubit(0,2), cirq.GridQubit(0,3)],
                [   cirq.GridQubit(1,0), cirq.GridQubit(1,1), cirq.GridQubit(1,2), cirq.GridQubit(1,3),
                   cirq.GridQubit(2,0), cirq.GridQubit(2,1), cirq.GridQubit(2,2), cirq.GridQubit(2,3)],
                [   cirq.GridQubit(3,0), cirq.GridQubit(3,1), cirq.GridQubit(3,2), cirq.GridQubit(3,3)]],
            {   cirq.GridQubit(0,0):0, cirq.GridQubit(0,1):1, cirq.GridQubit(0,2):2, cirq.GridQubit(0,3):3,
                cirq.GridQubit(1,0):4, cirq.GridQubit(1,1):5, cirq.GridQubit(1,2):6, cirq.GridQubit(1,3):7,
                cirq.GridQubit(2,0):8, cirq.GridQubit(2,1):9, cirq.GridQubit(2,2):10, cirq.GridQubit(2,3):11,
                cirq.GridQubit(3,0):12, cirq.GridQubit(3,1):13, cirq.GridQubit(3,2):14, cirq.GridQubit(3,3):15},
        ),
    ]
)
def test_set_get_subsystem_qubit_map(subsystem_qubits, target_qubit_map):
    n = [1,2]
    j_v0 = np.ones((n[0]-1,n[1]))
    j_h0 = np.ones((n[0],n[1]-1))
    h0 = np.ones((n[0],n[1]))

    #This does not matter so much
    #We simply need a SpinModel like object
    ising = Ising("GridQubit", n, j_v0, j_h0, h0, "Z")

    ising.subsystem_qubits = subsystem_qubits
    test_qubit_map = ising.basics.get_subsystem_qubit_map(ising)
    assert(target_qubit_map == test_qubit_map)

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