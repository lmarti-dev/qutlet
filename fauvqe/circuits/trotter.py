"""
    This should implement Trotterization of any cirq.PauliSum Hamiltonian
    or Hamiltonian function hamiltonian(t) that returns cirq.PauliSum 

    TODO s:
        -allow for t0 != 0
"""
import cirq
import numpy as np

from numbers import Real

def set_circuit(self):
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
    hamiltonian = self.trotter.options.get("hamiltonian")
    t0 = self.trotter.options.get("t0")
    tf = self.trotter.options.get("tf")
    q = self.trotter.options.get("trotter_order")
    m = self.trotter.options.get("trotter_number")

    #assert Trotter order and Trotter number are Integers
    assert isinstance(q, int), "Trotter order q must be Integer. Received: {}".format(q)
    assert isinstance(m, int), "Trotter number m must be Integer. Received: {}".format(m)

    if isinstance(hamiltonian, cirq.PauliSum):
        #print("Time-independent Trotterization")
        _circuit = m * self.trotter.get_step(self, hamiltonian, tf/m, q)
    else:
        #print("Time-dependent Trotterization")
        _circuit = cirq.Circuit()
        for i_m in range(m):
            #For q != 1 this is possibly still wrong
            _circuit.append(self.trotter.get_step(self, hamiltonian(float((tf/m)*(i_m + 0.5))), float(tf/m), q))

    if self.trotter.options.get('return'):
        return _circuit
    elif self.trotter.options.get('append'):
        self.circuit.append(_circuit)
    else:
        self.circuit = _circuit

    
def get_step(   self, 
                hamiltonian: cirq.PauliSum, 
                t: np.uint, 
                q: np.uint) -> cirq.Circuit:
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
        q=1: \prod_k e^(-i t/m* h_k(t/m)): cirq.Circuit
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
        partone = self.trotter.get_step(self, hamiltonian, nu*t, q-2)
        parttwo = self.trotter.get_step(self, hamiltonian, (1-4*nu)*t, q-2)
        return 2*partone + parttwo + 2*partone
    else:
        raise NotImplementedError()
    
def _first_order_trotter_circuit(   self, 
                                    hamiltonian: cirq.PauliSum, 
                                    t: Real) -> cirq.Circuit:
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
        Circuit describing Trotterized Time Evolution
    """
    return cirq.Circuit(*[self.trotter._get_gate_from_paulistring(self, pstr, hamiltonian, t) for pstr in hamiltonian._linear_dict])

def _get_gate_from_paulistring( self,
                                pauli_string: frozenset,
                                hamiltonian: cirq.PauliSum, 
                                t: Real):
    #Try to just return appropiate gates
    _paulis=[]
    _qubits =[]
    _exponent=2/np.pi *float(t) * float(np.real(hamiltonian._linear_dict[pauli_string]))
    temp=1
    for item in pauli_string:
        temp = temp * item[1](item[0])
        _qubits.append(item[0])
        _paulis.append(item[1])
        
    if len(_paulis) > 2:
        return temp**np.real(_exponent)
    elif len(_paulis) == 2:
        if _paulis[0] != _paulis[1]:
            return temp**np.real(_exponent)
        elif _paulis[0] == cirq.Z:
            return cirq.ZZPowGate(exponent=_exponent).on(*_qubits)
        elif _paulis[0] == cirq.X:
            return cirq.XXPowGate(exponent=_exponent).on(*_qubits)
        else:
            return cirq.YYPowGate(exponent=_exponent).on(*_qubits)
    elif len(_paulis) == 1:
        if _paulis[0] == cirq.X:
            return cirq.XPowGate(exponent=_exponent).on(*_qubits)
        elif _paulis[0] == cirq.Z:
            return cirq.ZPowGate(exponent=_exponent).on(*_qubits)
        else:
            return cirq.YPowGate(exponent=_exponent).on(*_qubits)
    else:
        return cirq.IdentityGate

def get_parameters(self, name: str='', delim: str = ','):
    #TODO extend this to time dependent models
    if(name == ''):
        if(self.trotter.options.get("trotter_order") > 2):
            raise NotImplementedError
        t_number = self.trotter.options.get("trotter_number")
        parameters = []
        for m in range(t_number):
            #TODO Replace this with self.hamiltonian(tf)?
            for pstr in self._hamiltonian._linear_dict:
                #print(self.hamiltonian._linear_dict[pstr])
                parameters.append(self._hamiltonian._linear_dict[pstr] / t_number)
        parameters = np.array(parameters) * np.real(2/np.pi * self.t)
        if(self.trotter.options.get("trotter_order")==2):
            return 0.5*np.concatenate((parameters, parameters[-1::-1]))
        else:
            return parameters
    else:
        return np.genfromtxt(name, delimiter=delim) #pragma: no cover