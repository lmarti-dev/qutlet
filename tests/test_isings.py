# external imports
import numpy as np
import cirq

# internal imports
from fauvqe import Ising


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
        wf_energy = ising_obj.energy(wf)
        assert (
            abs(E_exp - wf_energy) < self.atol
        ), "Simple JZZ hX energy test failed; expected: {}, received {}, tolerance {}".format(
            E_exp, wf_energy, self.atol
        )

    def simple_energy_JZZ_hZ_test(self, qubittype, n, j_v, j_h, h, test_gate, E_exp):
        # Create Ising object
        ising_obj = Ising(qubittype, n, j_v, j_h, h)

        # Apply test gate
        for i in range(ising_obj.n[0]):
            for j in range(ising_obj.n[1]):
                ising_obj.circuit.append(test_gate(ising_obj.qubits[i][j]))

        # Simulate
        wf = ising_obj.simulator.simulate(ising_obj.circuit).state_vector()

        # Renormalise wavefunction as required by qsim:
        wf = wf / np.sqrt(abs(wf.dot(np.conj(wf))))

        # Test where calculated energy fits to energy expectation E_exp
        wf_energy = ising_obj.energy(wf, field="Z")
        assert (
            abs(E_exp - wf_energy) < self.atol
        ), "Simple JZZ hZ energy test failed; expected: {}, received {}, tolerance {}".format(
            E_exp, wf_energy, self.atol
        )

    # add the option to apply not to all qubits
    def simple_spin_value_map_test(self, qubittype, n, j_v, j_h, h, test_gate, vm_exp, app_to=[]):
        # Check wether wm_exp is a dictionary
        assert dict == type(
            vm_exp
        ), "Ising, value map test failed: vm_exp expected to be dictionary, received: {}".format(
            type(vm_exp)
        )

        if np.size(app_to) == 0:
            app_to = np.ones(n)

        # Create Ising object
        ising_obj = Ising(qubittype, n, j_v, j_h, h)

        # Dummy to generate 'empty circuit'
        for i in range(ising_obj.n[0]):
            for j in range(ising_obj.n[1]):
                ising_obj.circuit.append(cirq.Z(ising_obj.qubits[i][j]) ** 2)

        # Apply test gate
        for i in range(ising_obj.n[0]):
            for j in range(ising_obj.n[1]):
                if app_to[i][j] == 1:
                    ising_obj.circuit.append(test_gate(ising_obj.qubits[i][j]))

        # Simulate
        wf = ising_obj.simulator.simulate(ising_obj.circuit).state_vector()

        # Renormalise wavefunction as required by qsim:
        wf = wf / np.sqrt(abs(wf.dot(np.conj(wf))))

        # Test where calculated spin value map fits to expectation spin
        # value map dictionary vm_exp
        value_map = ising_obj.get_spin_vm(wf)
        assert len(value_map) == len(
            vm_exp
        ), "Ising, value map test failed: length expected: {}, received: {}".format(
            len(vm_exp), len(value_map)
        )
        # If elements in value_map and vm_exp do not match receive KeyError and test fails
        for element in value_map:
            assert np.allclose(
                value_map[element], vm_exp[element], rtol=0, atol=self.atol
            ), "Ising, value map test failed: for element {}, expected: {}, received: {}".format(
                element, vm_exp[element], value_map[element]
            )
