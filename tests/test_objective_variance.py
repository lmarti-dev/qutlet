from mimetypes import init
import cirq
import numpy as np
import pytest
from random import randrange

from fauvqe import AbstractModel, ANNNI, Converter, ExpectationValue, Ising, Variance

class MockAbstractModel(AbstractModel):
    def copy(self):
        return MockAbstractModel()

    def energy(self):
        return np.array([])

    def from_json_dict(self, dct):
        return MockInitialiser()

    def to_json_dict(self):
        return {}

    def _set_hamiltonian(self, reset: bool = True):
        self._hamiltonian = cirq.PauliSum()

@pytest.mark.parametrize(
    "state, observables, variances, n",
    [
        # 1 Qubit
        (
            np.array([1,0]).astype(np.complex128),
            cirq.PauliSum.from_pauli_strings([cirq.Z(cirq.LineQubit(i)) for i in range(1)]),
            0,
            1,
        ),
        (
            np.array([1,0]).astype(np.complex128),
            cirq.PauliSum.from_pauli_strings([cirq.X(cirq.LineQubit(i)) for i in range(1)]),
            1,
            1,
        ),
        (
            None,
            cirq.PauliSum.from_pauli_strings([cirq.Y(cirq.LineQubit(i)) for i in range(1)]),
            1,
            1,
        ),
        (
            np.array([1,0]).astype(np.complex128),
            cirq.PauliSum.from_pauli_strings([cirq.X(cirq.LineQubit(0)), cirq.Z(cirq.LineQubit(0))]),
            1,
            1,
        ),
        # 1 Qubit & List of observables
        (
            np.array([1,0]).astype(np.complex128),
            [cirq.PauliSum.from_pauli_strings([cirq.Z(cirq.LineQubit(i)) for i in range(1)]),
            cirq.PauliSum.from_pauli_strings([cirq.X(cirq.LineQubit(i)) for i in range(1)]),
            cirq.PauliSum.from_pauli_strings([cirq.Y(cirq.LineQubit(i)) for i in range(1)])],
            [0,1,1 ],
            1
        ),
        # 2 Qubits
        (
            np.array([0.5,0.5, 0.5, 0.5]).astype(np.complex128),
            cirq.PauliSum.from_pauli_strings([cirq.Z(cirq.LineQubit(i)) for i in range(2)]),
            2,
            2,
        ),
        (
            np.array([0.5,0.5, 0.5, 0.5]).astype(np.complex128),
            cirq.PauliSum.from_pauli_strings([cirq.X(cirq.LineQubit(i)) for i in range(2)]),
            0,
            2,
        ),
        (
            np.array([0.5,0.5, 0.5, 0.5]).astype(np.complex128),
            cirq.PauliSum.from_pauli_strings([cirq.Y(cirq.LineQubit(i)) for i in range(2)]),
            2,
            2,
        ),
    ],
)
def test_evaluate_simple(state, observables, variances, n):
    model = MockAbstractModel("LineQubit", n)
    model.set_simulator("cirq")
    variance_obj = Variance(model, observables, state)

    #print("Observable(s): {}".format(observables))

    _qubit_map = {cirq.LineQubit(i): i for i in range(n)}
    _tmp=variance_obj.evaluate(_qubit_order=_qubit_map) 

    #print("variance_obj.evaluate(_qubit_order={}):\n{}".format(_qubit_map, _tmp))
    #print("variances: \t{}\n".format(variances))

    np.testing.assert_allclose(_tmp, variances, rtol=1e-14, atol=0)

@pytest.mark.parametrize(
    "n, j_v, j_h , h, field, init_state, basics_options, variances",
    [
        #Test Z systems
        #n0
        (
            [1,2],
            2*(np.random.rand(1-1,2)- 0.5),
            2*(np.random.rand(1,2-1)- 0.5),
            np.zeros((1,2)),
            "Z",
            randrange(2**2),
            {"append": False, "start": "identity"},
            0
        ),
        #n1
        (
            [2,2],
            2*(np.random.rand(2-1,2)- 0.5),
            2*(np.random.rand(2,2-1)- 0.5),
            np.zeros((2,2)),
            "Z",
            randrange(2**4),
            {"append": False, "start": "identity"},
            0
        ),
        #n2
        (
            [5,1],
            2*(np.random.rand(5-1,1)- 0.5),
            2*(np.random.rand(5,1-1)- 0.5),
            np.zeros((5,1)),
            "Z",
            randrange(2**5),
            {"append": False, "start": "identity"},
            0
        ),
        #n3
        (
            [3,2],
            2*(np.random.rand(3,2)- 0.5),
            2*(np.random.rand(3,2-1)- 0.5),
            np.zeros((3,2)),
            "Z",
            randrange(2**6),
            {"append": False, "start": "identity"},
            0
        ),
        #Test X systems
        #n4
        (
            [1,2],
            np.zeros((1,2)).astype(np.float64),
            np.zeros((1,2)).astype(np.float64),
            2*(np.random.rand(1,2).astype(np.float64)- 0.5),
            "X",
            randrange(2**2),
            {"append": False, "start": "hadamard"},
            0
        ),
        #n5
        (
            [2,2],
            np.zeros((2,2)),
            np.zeros((2,2)),
            2*(np.random.rand(2,2)- 0.5),
            "X",
            randrange(2**4),
            {"append": False, "start": "hadamard"},
            0
        ),
        #n6
        (
            [1,5],
            np.zeros((1,5)),
            np.zeros((1,5)),
            2*(np.random.rand(1,5)- 0.5),
            "X",
            randrange(2**5),
            {"append": False, "start": "hadamard"},
            0
        ),
        #n7
        (
            [2,3],
            np.zeros((2,3)),
            np.zeros((2,3)),
            2*(np.random.rand(2,3)- 0.5),
            "X",
            randrange(2**6),
            {"append": False, "start": "hadamard"},
            0
        ),
        #Test Eigen basis
        #n8
        (
            [1,2],
            2*(np.random.rand(1-1,2)- 0.5),
            2*(np.random.rand(1,2-1)- 0.5),
            2*(np.random.rand(1,2)- 0.5),
            "X",
            randrange(2**2),
            {"append": False, "start": "exact", "n_exact": [1,2], "b_exact": [1,1]},
            0
        ),
        # This fails but should not?
        #(
        #    [2,2],
        #    2*(np.random.rand(2-1,2)- 0.5),
        #    2*(np.random.rand(2,2-1)- 0.5),
        #    2*(np.random.rand(2,2)- 0.5),
        #    "X",
        #    randrange(2**4),
        #    {"append": False, "start": "exact", "n_exact": [2,2], "b_exact": [1,1]},
        #    0
        #),
        #n9
        (
            [2,2],
            np.ones((1,2)),
            np.ones((2,1)),
            np.ones((2,2)),
            "X",
            randrange(2**4),
            {"append": False, "start": "exact", "n_exact": [2,2], "b_exact": [1,1]},
            0
        ),
        #n10
        (
            [1,3],
            2*(np.random.rand(1-1,3)- 0.5),
            2*(np.random.rand(1,3)- 0.5),
            2*(np.random.rand(1,3)- 0.5),
            "X",
            randrange(2**3),
            {"append": False, "start": "exact", "n_exact": [1,3], "b_exact": [1,0]},
            0
        ),
        #n11
        (
            [5,1],
            2*(np.random.rand(5,1)- 0.5),
            2*(np.random.rand(5,1-1)- 0.5),
            2*(np.random.rand(5,1)- 0.5),
            "X",
            randrange(2**2),
            {"append": False, "start": "exact", "n_exact": [5,1], "b_exact": [0,1]},
            0
        ),
        #n12
        (
            [2,3],
            2*(np.random.rand(2-1,3)- 0.5),
            2*(np.random.rand(2,3)- 0.5),
            2*(np.random.rand(2,3)- 0.5),
            "X",
            randrange(2**6),
            {"append": False, "start": "exact", "n_exact": [2,3], "b_exact": [1,0]},
            0
        ),
        #n13
        (
            [2,3],
            np.ones((1,3)),
            np.ones((2,2)),
            np.ones((2,3)),
            "X",
            0,
            {"append": False, "start": "exact", "n_exact": [2,3], "b_exact": [1,1]},
            0,
        ),
        #n14
        (
            [2,3],
            np.ones((1,3)),
            np.ones((2,2)),
            np.ones((2,3)),
            "Z",
            None,
            {"append": False},
            0,
        ),
    ],
)
def test_evaluate_Ising(n, j_v, j_h , h, field, init_state, basics_options, variances):
    model = Ising("GridQubit", n, j_v, j_h , h , field)
    model.set_simulator("cirq", {"dtype": np.complex128})
    model.set_circuit("basics", basics_options)

    if init_state is not None:
        state=model.simulator.simulate( model.circuit, 
                                        initial_state=init_state).state_vector()
        variance_obj = Variance(model, wavefunction=state)
    else:
        variance_obj = Variance(model)

    np.testing.assert_allclose(variances, variance_obj.evaluate(atol=1e-14), rtol=1e-14, atol=1e-13)

@pytest.mark.parametrize(
    "n, j_v, j_h, h, basics_options1, basics_options2",
    [
        (
            #basics_options1: X/Z partition of ZZ-X Ising model
            #basics_options2: Parition in 2 1x2 subsystems where h = h/2
            [2,2],
            np.ones((2-1,2)),
            np.ones((2,2-1)),
            np.ones((2,2)),
            {   "start": "exact",
                "subsystem_diagonalisation": False,
                "subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(1,0), cirq.GridQubit(1,1)], 
                                    [cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
                "subsystem_j_v": [  0*np.transpose(np.array( [[[1]], [[1]]]), (1,0, 2)),
                                    np.transpose(np.array([[[1]], [[1]]]), (1,0, 2))],
                "subsystem_j_h": [  0*np.transpose(np.array([[[1], [1]]]), (1,0, 2)),
                                    np.transpose(np.array([[[1], [1]]]), (1,0, 2))],
                "subsystem_h": [    np.transpose(np.array([[[1], [1]], [[1], [1]]]), (1,0, 2)), 
                                    0*np.transpose(np.array([[[1], [1]], [[1], [1]]]), (1,0, 2)) ]
            },
            {   "start": "exact",
                "subsystem_diagonalisation": False,
                "subsystem_qubits": [[cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(1,0), cirq.GridQubit(1,1)], 
                                    [cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(1,0), cirq.GridQubit(1,1)]],
                "subsystem_j_v": [  np.transpose(np.array( [[[1]], [[1]]]), (1,0, 2)),
                                    0*np.transpose(np.array([[[1]], [[1]]]), (1,0, 2))],
                "subsystem_j_h": [  0*np.transpose(np.array([[[1], [1]]]), (1,0, 2)),
                                    np.transpose(np.array([[[1], [1]]]), (1,0, 2))],
                "subsystem_h": [    0.5*np.transpose(np.array([[[1], [1]], [[1], [1]]]), (1,0, 2)), 
                                    0.5*np.transpose(np.array([[[1], [1]], [[1], [1]]]), (1,0, 2)) ]
            },
        ),
        (
            #basics_options1: X/Z partition of ZZ-X Ising model
            #basics_options2: Parition in 1x3 and 2x1  subsystems where h = h/2
            [2,3],
            np.ones((2-1,3)),
            np.ones((2,3-1)),
            np.ones((2,3)),
            {   "start": "exact",
                "subsystem_diagonalisation": False,
                "subsystem_qubits": [[  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2), 
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2)], 
                                    [  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2), 
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2)]],
                "subsystem_j_v": [  0*np.transpose(np.array([[[1], [1], [1]]]), (0,1, 2)),
                                    np.transpose(np.array([[[1], [1], [1]]]), (0,1, 2))],
                "subsystem_j_h": [  0*np.transpose(np.array( [[[1], [1]], [[1], [1]]]), (1,0, 2)),
                                    np.transpose(np.array([[[1], [1]], [[1], [1]]]), (1,0, 2))],
                "subsystem_h": [    np.transpose(np.array([[[1], [1], [1]], [[1], [1], [1]]]), (0, 1,2)), 
                                    0*np.transpose(np.array([[[1], [1], [1]], [[1], [1], [1]]]), (0,1, 2)) ]
            },
            {   "start": "exact",
                "subsystem_diagonalisation": False,
                "subsystem_qubits": [[  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2), 
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2)], 
                                    [  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2), 
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2)]],
                "subsystem_j_v": [  np.transpose(np.array([[[1], [1], [1]]]), (0,1, 2)),
                                    0*np.transpose(np.array([[[1], [1], [1]]]), (0,1, 2))],
                "subsystem_j_h": [  0*np.transpose(np.array( [[[1], [1]], [[1], [1]]]), (1,0, 2)),
                                    np.transpose(np.array([[[1], [1]], [[1], [1]]]), (1,0, 2))],
                "subsystem_h": [    0.5*np.transpose(np.array([[[1], [1], [1]], [[1], [1], [1]]]), (0, 1,2)), 
                                    0.5*np.transpose(np.array([[[1], [1], [1]], [[1], [1], [1]]]), (0,1, 2)) ]
            },
        ),
        (
            #basics_options1: X/Z partition of ZZ-X Ising model
            #basics_options2: Parition in 1x4 and 2x1 subsystems where h = h/2
            [2,4],
            np.ones((2-1,4)),
            np.ones((2,4-1)),
            np.ones((2,4)),
            {   "start": "exact",
                "subsystem_diagonalisation": False,
                "subsystem_qubits": [[  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2),cirq.GridQubit(0,3),
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2),cirq.GridQubit(1,3)], 
                                    [  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2),cirq.GridQubit(0,3),
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2),cirq.GridQubit(1,3)]],
                "subsystem_j_v": [  0*np.transpose(np.array([[[1], [1], [1], [1]]]), (0,1, 2)),
                                    np.transpose(np.array([[[1], [1], [1], [1]]]), (0,1, 2))],
                "subsystem_j_h": [  0*np.transpose(np.array( [[[1], [1]], [[1], [1]], [[1], [1]]]), (1,0, 2)),
                                    np.transpose(np.array([[[1], [1]], [[1], [1]], [[1], [1]]]), (1,0, 2))],
                "subsystem_h": [    np.transpose(np.array([[[1], [1], [1], [1]], [[1], [1], [1], [1]]]), (0, 1,2)), 
                                    0*np.transpose(np.array([[[1], [1], [1], [1]], [[1], [1], [1], [1]]]), (0,1, 2)) ]
            },
            {   "start": "exact",
                "subsystem_diagonalisation": False,
                "subsystem_qubits": [[  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2),cirq.GridQubit(0,3),
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2),cirq.GridQubit(1,3)], 
                                    [  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2),cirq.GridQubit(0,3),
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2),cirq.GridQubit(1,3)]],
                "subsystem_j_v": [  0*np.transpose(np.array([[[1], [1], [1], [1]]]), (0,1, 2)),
                                    np.transpose(np.array([[[1], [1], [1], [1]]]), (0,1, 2))],
                "subsystem_j_h": [  np.transpose(np.array( [[[1], [1]], [[1], [1]], [[1], [1]]]), (1,0, 2)),
                                    0*np.transpose(np.array([[[1], [1]], [[1], [1]], [[1], [1]]]), (1,0, 2))],
                "subsystem_h": [    0.5*np.transpose(np.array([[[1], [1], [1], [1]], [[1], [1], [1], [1]]]), (0, 1,2)), 
                                    0.5*np.transpose(np.array([[[1], [1], [1], [1]], [[1], [1], [1], [1]]]), (0,1, 2)) ]
            },
        ),
        (
            #basics_options1: X/Z partition of ZZ-X Ising model
            #basics_options2: Parition in 2 2x2 and 2x1, 2x2, 2x1 subsystem
            #                   where h = h/2 and j_v=j_v/2
            [2,4],
            np.ones((2-1,4)),
            np.ones((2,4-1)),
            np.ones((2,4)),
            {   "start": "exact",
                "subsystem_diagonalisation": False,
                "subsystem_qubits": [[  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2),cirq.GridQubit(0,3),
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2),cirq.GridQubit(1,3)], 
                                    [  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2),cirq.GridQubit(0,3),
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2),cirq.GridQubit(1,3)]],
                "subsystem_j_v": [  0*np.transpose(np.array([[[1], [1], [1], [1]]]), (0,1, 2)),
                                    np.transpose(np.array([[[1], [1], [1], [1]]]), (0,1, 2))],
                "subsystem_j_h": [  0*np.transpose(np.array( [[[1], [1]], [[1], [1]], [[1], [1]]]), (1,0, 2)),
                                    np.transpose(np.array([[[1], [1]], [[1], [1]], [[1], [1]]]), (1,0, 2))],
                "subsystem_h": [    np.transpose(np.array([[[1], [1], [1], [1]], [[1], [1], [1], [1]]]), (0, 1,2)), 
                                    0*np.transpose(np.array([[[1], [1], [1], [1]], [[1], [1], [1], [1]]]), (0,1, 2)) ]
            },
            {   "start": "exact",
                "subsystem_diagonalisation": False,
                "subsystem_qubits": [[  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2),cirq.GridQubit(0,3),
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2),cirq.GridQubit(1,3)], 
                                    [  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2),cirq.GridQubit(0,3),
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2),cirq.GridQubit(1,3)]],
                "subsystem_j_v": [  0.5*np.transpose(np.array([[[1], [1], [1], [1]]]), (0,1, 2)),
                                    0.5*np.transpose(np.array([[[1], [1], [1], [1]]]), (0,1, 2))],
                "subsystem_j_h": [  np.transpose(np.array( [[[1], [1]], [[0], [0]], [[1], [1]]]), (1,0, 2)),
                                    np.transpose(np.array([[[0], [0]], [[1], [1]], [[0], [0]]]), (1,0, 2))],
                "subsystem_h": [    0.5*np.transpose(np.array([[[1], [1], [1], [1]], [[1], [1], [1], [1]]]), (0, 1,2)), 
                                    0.5*np.transpose(np.array([[[1], [1], [1], [1]], [[1], [1], [1], [1]]]), (0,1, 2)) ]
            },
        ),
        # Note for 2 x 4
        #Fails basics_options1: Parition in 1x4 and 2x1 subsystems where h = h/2 (previous basics_options2)
        #Fails basics_options2: Parition in 2 2x2 and 2x1, 2x2, 2x1 subsystem
        #                   where h = h/2 and j_v=j_v/2
        #  But sum of variances for option 2 is always smaller
        #
    ],
)     
def test_evaluate_H_partitions(n, j_v, j_h, h,basics_options1, basics_options2):
    #Test here whether subsystems have smaller variance that X,Z for J=h=1
    ising1 = Ising("GridQubit", n, j_v, j_h, h, "X")

    ising2 = ising1.copy()
    ising1.set_circuit("basics", basics_options1)
    ising2.set_circuit("basics", basics_options2)

    #Assert that subsystem partition match Hamiltonian
    assert(ising1.hamiltonian() == sum(ising1.subsystem_hamiltonians))
    assert(ising1.hamiltonian() == sum(ising2.subsystem_hamiltonians))
    
    #Init variance object
    variance_obj = Variance(ising1)

    #Use 3 lowest energy eigenstates to confirm reduced variance
    #These are the relevant ones for the band gap
    #Also consider 2. Excited state as for J>h GS ~ 1.ES
    #Use converter + scipy sparse solver
    converter_obj = Converter()
    scipy_crsmatrix = converter_obj.cirq_paulisum2scipy_crsmatrix(ising1.hamiltonian() , dtype=np.float64)
    k_excited_states = 3
    ising1.diagonalise(solver = "scipy.sparse", 
                        solver_options= { "k": k_excited_states},
                        matrix=scipy_crsmatrix)
    ising1.eig_vec = ising1.eig_vec.astype(np.complex128)

    #Print outs also take time hence comment them out:
    #ising1.set_circuit("basics", {"append": False, "start": "hadamard"})
    #np.set_printoptions(precision=6, threshold=1024, linewidth= 150)
    #print(ising1.eig_val)
    #exp_obj=ExpectationValue(ising1)
    #for i in range(k_excited_states):
    #    #state_X = ising1.simulator.simulate(ising1.circuit, initial_state=ising1.eig_vec[:,i]).state_vector()
    #    print("{}. Excited state, Energy: {}".format(i, exp_obj.evaluate(ising1.eig_vec[:,i])))
    #    #print("Z basis\n{}\nX basis\n{}".format( ising1.eig_vec[:,i], state_X))
    #    print(type(ising1.eig_vec[0,i]))
    #    print("Variances partition 1: {}".format(variance_obj.evaluate(observables=ising1.subsystem_hamiltonians, wavefunction=ising1.eig_vec[:,i])))
    #    print("Variances partition 2: {}".format(variance_obj.evaluate(observables=ising2.subsystem_hamiltonians, wavefunction=ising1.eig_vec[:,i])))
    #    print("Variances full system: {}".format(variance_obj.evaluate(observables=ising1.hamiltonian(), wavefunction=ising1.eig_vec[:,i])))

    for i in range(k_excited_states):
        _tmp1 = np.sort(abs(np.array(variance_obj.evaluate(  observables=ising1.subsystem_hamiltonians, 
                                                            wavefunction=ising1.eig_vec[:,i]))))

        _tmp2 = np.sort(abs(np.array(variance_obj.evaluate(  observables=ising2.subsystem_hamiltonians, 
                                                            wavefunction=ising1.eig_vec[:,i]))))

        assert(all(_tmp1 > _tmp2))

@pytest.mark.higheffort
@pytest.mark.parametrize(
    "n, j_v, j_h, h, basics_options1, basics_options2",
    [
        (
        #basics_options1: X/Z partition of ZZ-X Ising model
        #basics_options2: Parition in 2 2x4 and 1x4, 2x4, 1x4 subsystem
        #                   where h = h/2 and j_v=j_v/2
        # This is currently to slow to run: figure out why
            [4,4],
            np.ones((4-1,4)),
            np.ones((4,4-1)),
            np.ones((4,4)),
            {   "start": "exact",
                "subsystem_diagonalisation": False,
                "subsystem_qubits": [[  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2),cirq.GridQubit(0,3),
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2),cirq.GridQubit(1,3),
                                        cirq.GridQubit(2,0), cirq.GridQubit(2,1),cirq.GridQubit(2,2),cirq.GridQubit(2,3),
                                        cirq.GridQubit(3,0),cirq.GridQubit(3,1), cirq.GridQubit(3,2),cirq.GridQubit(3,3)], 
                                    [  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2),cirq.GridQubit(0,3),
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2),cirq.GridQubit(1,3),
                                        cirq.GridQubit(2,0), cirq.GridQubit(2,1),cirq.GridQubit(2,2),cirq.GridQubit(2,3),
                                        cirq.GridQubit(3,0),cirq.GridQubit(3,1), cirq.GridQubit(3,2),cirq.GridQubit(3,3)]],
                "subsystem_j_v": [  0*np.transpose(np.array([ [[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]]]), (0,1, 2)),
                                    np.transpose(np.array([ [[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]]]), (0,1, 2))],
                "subsystem_j_h": [  0*np.transpose(np.array([ [[1], [1], [1], [1]], 
                                                            [[1], [1],[1], [1]], 
                                                            [[1], [1], [1], [1]]]), (1,0, 2)),
                                    np.transpose(np.array([ [[1], [1], [1], [1]], 
                                                            [[1], [1],[1], [1]], 
                                                            [[1], [1], [1], [1]]]), (1,0, 2))],
                "subsystem_h": [    np.transpose(np.array([[[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]],  
                                                            [[1], [1], [1], [1]]]), (0, 1,2)), 
                                    0*np.transpose(np.array([[[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]],  
                                                            [[1], [1], [1], [1]]]), (0, 1,2)) ]
            },
            {   "start": "exact",
                "subsystem_diagonalisation": False,
                "subsystem_qubits": [[  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2),cirq.GridQubit(0,3),
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2),cirq.GridQubit(1,3),
                                        cirq.GridQubit(2,0), cirq.GridQubit(2,1),cirq.GridQubit(2,2),cirq.GridQubit(2,3),
                                        cirq.GridQubit(3,0),cirq.GridQubit(3,1), cirq.GridQubit(3,2),cirq.GridQubit(3,3)], 
                                    [  cirq.GridQubit(0,0), cirq.GridQubit(0,1),cirq.GridQubit(0,2),cirq.GridQubit(0,3),
                                        cirq.GridQubit(1,0),cirq.GridQubit(1,1), cirq.GridQubit(1,2),cirq.GridQubit(1,3),
                                        cirq.GridQubit(2,0), cirq.GridQubit(2,1),cirq.GridQubit(2,2),cirq.GridQubit(2,3),
                                        cirq.GridQubit(3,0),cirq.GridQubit(3,1), cirq.GridQubit(3,2),cirq.GridQubit(3,3)]],
                "subsystem_j_v": [  0.5*np.transpose(np.array([ [[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]]]), (0,1, 2)),
                                    0.5*np.transpose(np.array([ [[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]]]), (0,1, 2))],
                #Note here exist entries !=1
                "subsystem_j_h": [  np.transpose(np.array([ [[1], [1], [1], [1]], 
                                                            [[0], [0], [0], [0]], 
                                                            [[1], [1], [1], [1]]]), (1,0, 2)),
                                    np.transpose(np.array([ [[0], [0], [0], [0]], 
                                                            [[1], [1], [1], [1]], 
                                                            [[0], [0], [0], [0]]]), (1,0, 2))],
                "subsystem_h": [    0.5*np.transpose(np.array([[[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]],  
                                                            [[1], [1], [1], [1]]]), (0, 1,2)), 
                                    0.5*np.transpose(np.array([[[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]],
                                                            [[1], [1], [1], [1]],  
                                                            [[1], [1], [1], [1]]]), (0, 1,2)) ]
            },
        ),
    ],
)
def test_evaluate_H_partitions_higheffort(n, j_v, j_h, h,basics_options1, basics_options2):
    #Test here whether subsystems have smaller variance that X,Z for J=h=1
    ising1 = Ising("GridQubit", n, j_v, j_h, h, "X")
    ising1.set_simulator("cirq")

    ising2 = ising1.copy()
    ising1.set_circuit("basics", basics_options1)
    ising2.set_circuit("basics", basics_options2)

    #Assert that subsystem partition match Hamiltonian
    assert(ising1.hamiltonian() == sum(ising1.subsystem_hamiltonians))
    assert(ising1.hamiltonian() == sum(ising2.subsystem_hamiltonians))
    
    #Init variance object
    variance_obj = Variance(ising1)

    #Use 3 lowest energy eigenstates to confirm reduced variance
    #These are the relevant ones for the band gap
    #Also consider 2. Excited state as for J>h GS ~ 1.ES
    #Use converter + scipy sparse solver
    converter_obj = Converter()
    scipy_crsmatrix = converter_obj.cirq_paulisum2scipy_crsmatrix(ising1.hamiltonian() , dtype=np.float64)
    k_excited_states = 3
    ising1.diagonalise(solver = "scipy.sparse", 
                        solver_options= { "k": k_excited_states},
                        matrix=scipy_crsmatrix)
    ising1.eig_vec = ising1.eig_vec.astype(np.complex128)

    #Print outs:
    #ising1.set_circuit("basics", {"append": False, "start": "hadamard"})
    #np.set_printoptions(precision=6, threshold=1024, linewidth= 150)
    #print(ising1.eig_val)
    #exp_obj=ExpectationValue(ising1)
    #for i in range(k_excited_states):
    #    #state_X = ising1.simulator.simulate(ising1.circuit, initial_state=ising1.eig_vec[:,i]).state_vector()
    #    print("{}. Excited state, Energy: {}".format(i, exp_obj.evaluate(ising1.eig_vec[:,i])))
    #    #print("Z basis\n{}\nX basis\n{}".format( ising1.eig_vec[:,i], state_X))
    #    print("Variances partition 1: {}".format(variance_obj.evaluate(observables=ising1.subsystem_hamiltonians, wavefunction=ising1.eig_vec[:,i])))
    #    print("Variances partition 2: {}".format(variance_obj.evaluate(observables=ising2.subsystem_hamiltonians, wavefunction=ising1.eig_vec[:,i])))
    #    print("Variances full system: {}".format(variance_obj.evaluate(observables=ising1.hamiltonian, wavefunction=ising1.eig_vec[:,i])))

    for i in range(k_excited_states):
        _tmp1 = np.sort(abs(np.array(variance_obj.evaluate(  observables=ising1.subsystem_hamiltonians, 
                                                            wavefunction=ising1.eig_vec[:,i]))))

        _tmp2 = np.sort(abs(np.array(variance_obj.evaluate(  observables=ising2.subsystem_hamiltonians, 
                                                            wavefunction=ising1.eig_vec[:,i]))))

        assert(all(_tmp1 > _tmp2))

@pytest.mark.parametrize(
    "model, atol",
    [
        (
            Ising(  "GridQubit", 
                    [1,2], 
                    2*(np.random.rand(1-1,2)- 0.5),
                    2*(np.random.rand(1,2-1)- 0.5),
                    np.zeros((1,2)), 
                    "Z"),
            1e-10
        ),
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    np.ones((1,2)),
                    np.ones((2,1)),
                    np.ones((2,2)), 
                    "X"),
            1e-10
        ),
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    np.ones((1,2)),
                    0.9*np.ones((2,1)),
                    np.ones((2,2)), 
                    "Z"),
            1e-10
        ),
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    np.ones((1,2)),
                    0.9*np.ones((2,1)),
                    np.ones((2,2)), 
                    "X"),
            1e-10
        ),
        # Currently this fails
        (
            Ising(  "GridQubit", 
                    [2,2], 
                    2*(np.random.rand(2-1,2)- 0.5),
                    2*(np.random.rand(2,2-1)- 0.5),
                    np.ones((2,2)), 
                    "X"),
            1e-10
        ),
        (
            Ising(  "GridQubit", 
                    [2,3], 
                    np.ones((1,3)),
                    np.ones((2,2)),
                    np.ones((2,3)), 
                    "X"),
            1e-10
        ),
        (
            Ising(  "GridQubit", 
                    [3,2], 
                    np.ones((2,2)),
                    np.ones((3,1)),
                    np.ones((3,2)), 
                    "X"),
            1e-10
        ),
        (
            Ising(  "GridQubit", 
                    [1,6], 
                    np.ones((0,6)),
                    np.ones((1,5)),
                    np.ones((1,6)), 
                    "X"),
            1e-10
        ),
        (
            Ising(  "GridQubit", 
                    [6,1], 
                    np.ones((5,1)),
                    np.ones((6,0)),
                    np.ones((6,1)), 
                    "X"),
            1e-10
        ),
    ],
)
def test_evaluate_eigenstates(model, atol):
    """
        If |ψ> is an eigenstate of H then:
            H ψ = E ψ
        Thus:
            Var_|ψ> (H) = <ψ| H²|ψ> - <ψ| H |ψ>² = E² - E² = 0 

        Note: this kind of test could potentially be more efficent by storing & loading the eigenstates
    """
    model.set_simulator("cirq", {"dtype": np.complex128})
    variance_obj = Variance(model)
    exp_obj=ExpectationValue(model)
    
    # If model is Ising or ANNNI, we can use sparse diagonalisation
    if False:#isinstance(model, Ising) or isinstance(model, ANNNI):
        converter_obj = Converter()
        scipy_crsmatrix = converter_obj.cirq_paulisum2scipy_crsmatrix(model.hamiltonian() , dtype=np.float64)
        model.diagonalise(solver = "scipy.sparse", 
                            solver_options= { "k": 3},
                            matrix=scipy_crsmatrix)
    else:
        model.diagonalise(solver_options= { "k": 3})

    print("Exact diagonalisation Es: \t{}".format(np.multiply(*model.n)*model.eig_val))
    print("Exact diagonalisation Es²: \t{}".format((np.multiply(*model.n)*model.eig_val)**2))

    # Copied from Variance class
    _qubit_order = {model.qubits[k][l]: int(l*model.n[0] + k) for l in range(model.n[1]) for k in range(model.n[0])}
    _qubit_order = {model.qubits[k][l]: int(k*model.n[1] + l) for l in range(model.n[1]) for k in range(model.n[0])}
    print("qubit_order: {}".format(_qubit_order))

    _tmp_E=[model.simulator.simulate_expectation_values(
                    program = cirq.Circuit(), # to keep previous behaviour use empty circuit
                    qubit_order = _qubit_order,
                    observables=model._hamiltonian, # This allows for a list of observables
                    initial_state=model.eig_vec[:,i],
                    ) for i in range(len(model.eig_vec[0,:]))]
    print("<ψ|H|ψ> from Variance: \t\t{}".format(_tmp_E))
    print("<ψ|H|ψ>² from Variance: \t{}".format(np.array(_tmp_E).T**2))

    _tmp_Eb=[exp_obj.evaluate(model.eig_vec[:,i]) for i in range(len(model.eig_vec[0,:]))]
    print("<ψ|H|ψ> from ExpectationValue: \t\t{}".format((np.multiply(*model.n)*np.array(_tmp_Eb))))
    print("<ψ|H|ψ>² from ExpectationValue: \t{}".format((np.multiply(*model.n)*np.array(_tmp_Eb))**2))
    print("|E_exact - <ψ|H|ψ> from ExpectationValue|: {}".format(np.multiply(*model.n)*abs(_tmp_Eb-model.eig_val)))

    _tmp_E2=[model.simulator.simulate_expectation_values(
                    program = cirq.Circuit(), # to keep previous behaviour use empty circuit
                    qubit_order = _qubit_order,
                    observables=model._hamiltonian**2, # This allows for a list of observables
                    initial_state=model.eig_vec[:,i],
                    ) for i in range(len(model.eig_vec[0,:]))]
    print("<ψ|H²|ψ>: \t\t\t{}".format(_tmp_E2))

    assert abs(variance_obj.evaluate(wavefunction=model.eig_vec[:,0])) < atol


def test_json():
    model = Ising("GridQubit", [2, 2], np.ones((1, 2)), np.ones((2, 1)), np.ones((2, 2)))
    model.set_simulator("cirq")
    
    observables=[cirq.X(cirq.GridQubit(0,0)), cirq.Y(cirq.GridQubit(0,1)), cirq.Z(cirq.GridQubit(1,0))*cirq.Z(cirq.GridQubit(1,1)) ]

    state=np.random.rand(2**4,1)
    state=state/np.linalg.norm(state)
    objective = Variance(model, observables= observables, wavefunction=state)

    json = objective.to_json_dict()    
    objective2 = Variance.from_json_dict(json)
    
    print(objective)

    assert (objective == objective2)

def test_repr():
    model = MockAbstractModel("LineQubit", 1)
    variance_obj = Variance(model,wavefunction=np.array([1,0]), observables=cirq.Z(cirq.LineQubit(0)))
    assert repr(variance_obj) == "<Variance observable={}>".format(cirq.Z(cirq.LineQubit(0)))
#############################################################
#                                                           #
#                    Assert tests                           #
#                                                           #
#############################################################
#def test_evaluate_assert():
#    model = MockAbstractModel("LineQubit", 1)
#    variance_obj = Variance(model,np.array([1,0]), cirq.Z(cirq.LineQubit(0)))
#    with pytest.raises(AssertionError):
#        variance_obj.evaluate() 