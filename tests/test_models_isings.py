# external imports
import numpy as np
import cirq

# internal imports
from fauvqe import Ising, ExpectationValue

class IsingTester:
    def __init__(self, atol):
        self.atol = atol

    def simple_energy_JZZ_hX_test(self, qubittype, n, j_v, j_h, h, test_gate, E_exp, basis):
        # Create Ising object
        ising_obj = Ising(qubittype, n, j_v, j_h, h)

        # Apply test gate
        for i in range(ising_obj.n[0]):
            for j in range(ising_obj.n[1]):
                if basis == "X":
                    # Add here cirq.H to transform in X eigenbasis
                    ising_obj.circuit.append(cirq.H(ising_obj.qubits[i][j]))
                ising_obj.circuit.append(test_gate(ising_obj.qubits[i][j]))

        # Simulate
        wf = ising_obj.simulator.simulate(ising_obj.circuit).state_vector()

        # Renormalise wavefunction as required by qsim:
        wf = wf / np.sqrt(abs(wf.dot(np.conj(wf))))

        # Test where calculated energy fits to energy expectation E_exp
        exp_val = ExpectationValue(ising_obj)
        wf_energy = exp_val.evaluate(wf)
        assert (
            abs(E_exp - wf_energy) < self.atol
        ), "Simple JZZ hX energy test failed; expected: {}, received {}, tolerance {}".format(
            E_exp, wf_energy, self.atol
        )

    def simple_energy_JZZ_hZ_test(self, qubittype, n, j_v, j_h, h, test_gate, E_exp):
        # Create Ising object
        ising_obj = Ising(qubittype, n, j_v, j_h, h, "Z")

        # Apply test gate
        for i in range(ising_obj.n[0]):
            for j in range(ising_obj.n[1]):
                ising_obj.circuit.append(test_gate(ising_obj.qubits[i][j]))

        # Simulate
        wf = ising_obj.simulator.simulate(ising_obj.circuit).state_vector()

        # Renormalise wavefunction as required by qsim:
        wf = wf / np.sqrt(abs(wf.dot(np.conj(wf))))

        # Test where calculated energy fits to energy expectation E_exp
        exp_val = ExpectationValue(ising_obj)
        wf_energy = exp_val.evaluate(wf)

        assert (
            abs(E_exp - wf_energy) < self.atol
        ), "Simple JZZ hZ energy test failed; expected: {}, received {}, tolerance {}".format(
            E_exp, wf_energy, self.atol
        )
    
    def compare_val_modulo_permutation(A, B, i):
        try:
            np.testing.assert_allclose(A[i], B[i], rtol=1e-14, atol=1e-14)
        except AssertionError:
            np.testing.assert_allclose(A[(i+1)%2], B[i], rtol=1e-14, atol=1e-14)

    def compare_vec_modulo_permutation(A, B, i):
        try:
            cirq.testing.lin_alg_utils.assert_allclose_up_to_global_phase(A[:,i], B[:,i], rtol=1e-14, atol=1e-14)
        except AssertionError:
            cirq.testing.lin_alg_utils.assert_allclose_up_to_global_phase(A[:,(i+1)%2], B[:,i], rtol=1e-14, atol=1e-14)