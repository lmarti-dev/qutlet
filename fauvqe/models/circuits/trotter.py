"""
    Include here methods that help generating Trotterized time evolutions
"""
from fauvqe.models.abstractmodel import AbstractModel

# external import
import numpy as np
import cirq
import sympy

def set_circuit(self):
    assert isinstance(self.trotter.options['q'], int), "Trotter order q must be Integer. Received: {}".format(q)
    assert isinstance(self.trotter.options['m'], int), "Trotter number m must be Integer. Received: {}".format(m)
    if not self.trotter.options['append']:
        self.circuit = cirq.Circuit()
    self.circuit.append(self.trotter.get_trotter_circuit_from_hamiltonian(self, self.hamiltonian, self.t, self.trotter.options['q'], self.trotter.options['m']))

def get_trotter_circuit_from_hamiltonian(self, hamiltonian, t, q, m):
    return m * self.get_single_step_trotter_circuit_from_hamiltonian(hamiltonian, t/m, q)
    
def get_single_step_trotter_circuit_from_hamiltonian(self, hamiltonian, t, q):
    if(q == 1):
        return self._first_order_trotter_circuit(hamiltonian, t)
    elif(q == 2):
        half = self._first_order_trotter_circuit(hamiltonian, 0.5*t)
        for k in range(len(half)-1, -1, -1):
            half.append(half[k])
        return half
    elif( (q % 2) == 0):
        nu = 1/(4 - 4**(1/(q - 1)))
        partone = self.get_single_step_trotter_circuit_from_hamiltonian(hamiltonian, nu*t, q-2)
        parttwo = self.get_single_step_trotter_circuit_from_hamiltonian(hamiltonian, (1-4*nu)*t, q-2)
        return 2*partone + parttwo + 2*partone
    else:
        raise NotImplementedError()
    
def _first_order_trotter_circuit(self, hamiltonian, t):
    """
    This function initialises the circuit for Trotter approximation.
    
    Parameters
    ----------
    hamiltonian: cirq.PauliSum
        System Hamiltonian
    t: float
        Total simulation time
    q: np.uint
        Order of Approximation
    m: np.uint
        Refinement of time R. t -> ( t/R )^R
    
    Returns
    ---------
    res: cirq.Circuit()
        Circuit describing Trotterized Time Evolution
    """
    res = cirq.Circuit()
    #Loop through all the addends in the PauliSum Hamiltonian
    for pstr in hamiltonian._linear_dict:
        #temp encodes each of the PauliStrings in the PauliSum hamiltonian which need to be turned into gates
        temp = 1
        #Loop through Paulis in the PauliString (pauli[1] encodes the cirq gate and pauli[0] encodes the qubit on which the gate acts)
        for pauli in pstr:
            temp = temp * pauli[1](pauli[0])
        #Append the PauliString gate in temp to the power of the time step * coefficient of said PauliString. The coefficient needs to be multiplied by a correction factor of 2/pi in order for the PowerGate to represent a Pauli exponential.
        res.append(temp**np.real(2/np.pi * t * hamiltonian._linear_dict[pstr]))
    return res