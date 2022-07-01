import cirq
import numpy as np

def set_circuit(self):
        hamiltonian = self.trotter.options.get('hamiltonian')
        tf = self.trotter.options.get('tf')
        q = self.trotter.options.get('trotter_order')
        m = self.trotter.options.get('trotter_number')

        _circuit = m * self.trotter.get_single_step_trotter_circuit_from_hamiltonian(self, hamiltonian, tf/m, q)

        if self.trotter.options.get('return'):
            return _circuit
        elif self.trotter.options.get('return'):
            self.circuit.append(_circuit)
        else:
            self.circuit = _circuit

    
def get_single_step_trotter_circuit_from_hamiltonian(self, hamiltonian, t, q):
    if(q == 1):
        return self.trotter._first_order_trotter_circuit(self, hamiltonian, t)
    elif(q == 2):
        half = self.trotter._first_order_trotter_circuit(self, hamiltonian, 0.5*t)
        for k in range(len(half)-1, -1, -1):
            half.append(half[k])
        return half
    elif( (q % 2) == 0):
        nu = 1/(4 - 4**(1/(q - 1)))
        partone = self.trotter.get_single_step_trotter_circuit_from_hamiltonian(self, hamiltonian, nu*t, q-2)
        parttwo = self.trotter.get_single_step_trotter_circuit_from_hamiltonian(self, hamiltonian, (1-4*nu)*t, q-2)
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
    #Copy the Trotter layer *m times.
    return res