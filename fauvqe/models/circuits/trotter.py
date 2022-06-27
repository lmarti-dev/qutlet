"""
This is a submodule for instances of AbstractModel and for the UtCost objective
Include here methods that help generating Trotterized time evolutions
"""
from fauvqe.models.abstractmodel import AbstractModel

# external import
import numpy as np
import cirq
from numbers import Real
import sympy

from fauvqe.models.spinModel import SpinModel

def set_circuit(self) -> None:
    """
        Sets the circuit for Trotter-Suzuki sequence of order q and with Trotter number m
        
        Parameters
        ----------
        self
        
        Returns
        -------
        void
    """
    #assert Trotter order and Trotter number are Integers
    assert isinstance(self.trotter.options['q'], int), "Trotter order q must be Integer. Received: {}".format(q)
    assert isinstance(self.trotter.options['m'], int), "Trotter number m must be Integer. Received: {}".format(m)
    #Reset circuit
    if not self.trotter.options['append']:
        self.circuit = cirq.Circuit()
    #Call helper function that generates a circuit from the Hamiltonian PauliSum
    self.circuit.append(self.trotter.get_trotter_circuit_from_hamiltonian(self, self.hamiltonian, self.t, self.trotter.options['q'], self.trotter.options['m']))

def get_trotter_circuit_from_hamiltonian(self, hamiltonian: cirq.PauliSum, t: Real, q: np.uint, m: np.uint) -> cirq.Circuit:
    """
        Sets the circuit for Trotter-Suzuki sequence of order q and with Trotter number m
        
        Parameters
        ----------
        self
        hamiltonian: cirq.PauliSum, 
            Hamiltonian that is trotterized.
            We iterate over the addends which are tuples of PauliStrings together with their interaction coefficients
        t: Real
            Simulation time
        q: np.uint
            Order of Approximation
        m: np.uint
            Refinement of time R. t -> ( t/R )^R
        
        Returns
        -------
        q=1: (\prod_k e^(-i t/m h_k) )^m: cirq.Circuit
        q=2: symmetrized
        q=2k: see https://arxiv.org/abs/1912.08854
    """
    return m * self.trotter.get_single_step_trotter_circuit_from_hamiltonian(self, hamiltonian, t/m, q)
    
def get_single_step_trotter_circuit_from_hamiltonian(self, hamiltonian: cirq.PauliSum, t: Real, q: np.uint) -> cirq.Circuit:
    """
        Sets the single time step t/m for Trotter-Suzuki sequence of order q. Different cases for order q:
        q=1: Linear Trotter-Suzuki
        q=2: Symmetrized gate ordering
        q=2k: Higher orderings, see https://arxiv.org/abs/1912.08854
        
        Parameters
        ----------
        self
        hamiltonian: cirq.PauliSum, 
            Hamiltonian that is trotterized.
            We iterate over the addends which are tuples of PauliStrings together with their interaction coefficients
        t: Real
            Simulation time
        q: np.uint
            Order of Approximation
        
        Returns
        -------
        q=1: \prod_k e^(-i t/m h_k): cirq.Circuit
    """
    if(q == 1):
        return self.trotter._first_order_trotter_circuit(self, hamiltonian, t)
    elif(q == 2):
        #forward order
        half = self.trotter._first_order_trotter_circuit(self, hamiltonian, 0.5*t)
        #backward order
        for k in range(len(half)-1, -1, -1):
            half.append(half[k])
        return half
    elif( (q % 2) == 0):
        nu = 1/(4 - 4**(1/(q - 1)))
        #recursive relation
        partone = self.trotter.get_single_step_trotter_circuit_from_hamiltonian(self, hamiltonian, nu*t, q-2)
        parttwo = self.trotter.get_single_step_trotter_circuit_from_hamiltonian(self, hamiltonian, (1-4*nu)*t, q-2)
        return 2*partone + parttwo + 2*partone
    else:
        raise NotImplementedError()
    
def _first_order_trotter_circuit(self, hamiltonian: cirq.PauliSum, t: Real) -> cirq.Circuit:
    """
    This function initialises the circuit for q=1 Trotter approximation.
    
    Parameters
    ----------
    hamiltonian: cirq.PauliSum
        System Hamiltonian
    t: float
        Total simulation time
    
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

def get_parameters(self, name: str='', delim: str = ','):
    if(name == ''):
        return np.genfromtxt(name, delimiter=delim)
    else:
        parameters = []
        for pstr in hamiltonian._linear_dict:
            parameters.append(pstr.coefficient)
        parameters = np.array(parameters) * np.real(2/np.pi * self.t)
        return parameters