"""
This is a submodule for CoolingModel()

This file is not exectuded, rather called within Cooling classes when:
-set_circuit('cooling') is called 

or functions are handed over to classical optimiser
"""
import numpy as np
import cirq
import sympy
import scipy
from typing import List
from numbers import Real
from itertools import chain
#import fauvqe.CoolingModel

def set_K(self, K: np.uint) -> None:
    """
        Sets the cooling gradation number K
        K=1: BangBang cooling
        K>1: LogSweep cooling
        
        Parameters
        ----------
        self
        K: np.uint
            New K to be set
        
        Returns
        -------
        void
    """
    if self.cooling.options["K"] != K:
        self.cooling.options["K"] = K
        self.cooling.options["append"] = False
        # Need to reset circuit
        self.cooling.set_circuit(self)

def set_circuit(self) -> None:
    """
        Sets the cooling citcuit
        K=1: BangBang cooling
        K>1: LogSweep cooling
        
        Parameters
        ----------
        self
        
        Returns
        -------
        void
    """
    K = self.cooling.options["K"]
    # Reset circuit
    if not self.cooling.options['append']:
        self.circuit = cirq.Circuit()
        self.circuit_param_values = np.zeros(1)
    if(K==1):
        self.circuit.append(self.cooling.BangBangProtocol(self))
    else:
        self.circuit.append(self.cooling.LogSweepProtocol(self))
    print("Circuit depth:", len(self.circuit))

#LogSweep
def LogSweepProtocol(self) -> None:
    """
        Sets the LogSweep cooling circuit
        
        Parameters
        ----------
        self
        
        Returns
        -------
        void
    """
    #code shorthands
    K = self.cooling.options["K"]
    m = self.cooling.options["m"]
    q = self.cooling.options["q"]
    self.m_sys.normalise()
    if(self.cooling.options["emin"] is None or self.cooling.options["emax"] is None):
        #Use optimal emax and emin if nothing is handed
        e_min, e_max, spectral_spread = self.cooling.__get_default_e_m(self)
        print("Using default values for emin: {} and emax: {} and spectral spread: {}".format(e_min, e_max, spectral_spread))
    else:
        e_min = self.cooling.options["emin"]
        e_max = self.cooling.options["emax"]
        spectral_spread = 2
    #Loop through gradation number K
    for k in range(K):
        #e: Energy gap of Two-Level system
        #gamma: Interaction constant
        #tau: Simulation time
        e, gamma, tau = self.cooling.__get_Log_Sweep_parameters(self, e_min, e_max, k)
        if(self.cooling.options["m"] is None):
            #Set Trotter number, if it is not handed over in options
            m = int(2*np.sqrt(1+ (spectral_spread**2)/(gamma**2)))
        #Reset ancilla energy gap
        self._set_h_anc(np.transpose([e/2 * np.ones((*self.m_anc.n,))], (1, 2, 0)))
        #Reset simulation time
        self.t = tau
        c = cirq.Circuit()
        if(self.cooling_type == "NA"):
            #Reset coupling constants...
            self._set_j_int(gamma/2 * self.j_int / self.j_int)
            #... and Hamiltonian
            self._set_hamiltonian()
            #Trotter Layer
            c.append( self.trotter.get_trotter_circuit_from_hamiltonian(self, self.hamiltonian, self.t, q, m) )
            #Reset Layer
            c.append( self.cooling._reset_layer(self) )
            yield c * self.cooling.options["time_steps"]
        elif(self.cooling_type == "1A"):
            for n0 in range(self.m_sys.n[0]):
                for n1 in range(self.m_sys.n[1]):
                    j_int = np.zeros(shape=(1,*self.m_sys.n))
                    j_int[0, n0, n1] = gamma/2
                    #Reset coupling constants...
                    self._set_j_int(j_int)
                    for pauli in [cirq.X, cirq.Y, cirq.Z]:
                        self._two_q_gates[self.nbr_2Q_sys + self.nbr_2Q_anc:self.nbr_2Q] = [lambda q1, q2: pauli(q1)*cirq.X(q2)]
                        #--- and Hamiltonian
                        self._set_hamiltonian()
                        #Trotter Layer
                        c.append( self.trotter.get_trotter_circuit_from_hamiltonian(self, self.hamiltonian, self.t, q, m) )
                        #Reset Layer
                        c.append( self.cooling._reset_layer(self) )
            yield c * self.cooling.options["time_steps"]
        
def __get_default_e_m(self) -> List[Real]:
    """
        Sets the optimal parameters for emin, emax and the spectral spread
        
        Parameters
        ----------
        self
        
        Returns
        -------
        e_min: Real
            Energy Gap
        e_max: Real
            Orthogonal norm of commutator between system Hamiltonian and system part of interaction Pauli
        spectral_spread:
            highest energy - ground state energy
    """
    _n = np.size(self.m_sys.qubits)
    _N = 2**_n
    if np.size(self.m_sys.eig_val) != _N or (np.shape(self.m_sys.eig_vec) != np.array((_N, _N)) ).all():
        self.m_sys.diagonalise(solver = "scipy", solver_options={"subset_by_index": [0, _N - 1]})
    energy_ex = self.m_sys.eig_val[0]
    energy_ex2 = self.m_sys.eig_val[1]
    e_min = (energy_ex2 - energy_ex)
    spectral_spread = (self.m_sys.eig_val[-1] - energy_ex)
    e_max = max( self.cooling.orth_norm(1.j*
        self.cooling.commutator(
            np.array(pauli(self.m_sys.qubits[i][j]).matrix(self.cooling.flatten(self.m_sys.qubits))),
            np.array(self.m_sys.hamiltonian.matrix())
        )) for pauli in [cirq.X, cirq.Y, cirq.Z] for i in range(self.m_sys.n[0]) for j in range(self.m_sys.n[1]) )
    return e_min, e_max, spectral_spread

def __get_Log_Sweep_parameters(self, e_min: Real, e_max: Real, k: np.uint) -> List[Real]:
    """
        Sets the LogSweep parameters e, gamma and tau
        
        Parameters
        ----------
        self
        e_min: Real
            Energy Gap
        e_max: Real
            orth_norm([H_S, V_S])
        k: np.uint
            index of gradation sweep
        
        Returns
        -------
        e: Real
            Two-Level gap
        gamma: Real
            Coupling constant
        tau:
            Simulation time
    """
    K = self.cooling.options["K"]
    #kth element of LogSweep through energies
    e = e_min**(k/(K-1)) * e_max**(1 - k/(K-1)) #Free Spin precession
    #Optimal delta proven in QDC paper: https://arxiv.org/pdf/1909.10538.pdf 
    h = e_max / e_min
    R = np.log(h) * ((1 - h) / (2 * (1 + h)) + np.log(2 * h / (1 + h)))
    delta_factor = np.log(K * 8 / R) / 2 / np.pi
    delta = delta_factor * e * (1 - 2 / (1 + (e_min / e_max)**(1 / (1 - K))))
    tau = 1/delta
    #gamma set by Fermi's golden rule
    gamma = np.pi / tau
    return e, gamma, tau

#BangBang
def BangBangProtocol(self) -> None:
    """
        Sets the BangBang cooling circuit
        
        Parameters
        ----------
        self
        
        Returns
        -------
        void
    """
    #Same parameters as in LogSweep, we need to loop through the three different cooling objects, however
    ex, gx, tx = self.cooling.__get_Bang_Bang_parameters(self, cirq.X)
    ey, gy, ty = self.cooling.__get_Bang_Bang_parameters(self, cirq.Y)
    ez, gz, tz = self.cooling.__get_Bang_Bang_parameters(self, cirq.Z)
    cool_x = self.cooling.__config_system(self.copy(), ex, gx, tx, cirq.X)
    cool_y = self.cooling.__config_system(self.copy(), ey, gy, ty, cirq.Y)
    cool_z = self.cooling.__config_system(self.copy(), ez, gz, tz, cirq.Z)
    
    if(self.cooling.options["m"] is None):
        m = int( 10*max([tx, ty, tz]))
    else:
        m = self.cooling.options["m"]
    q = self.cooling.options["q"]
    
    c = cirq.Circuit()
    if(self.cooling_type == "NA"):
        for system in [cool_x, cool_y, cool_z]:
            #Trotter Layer
            c.append( system.trotter.get_trotter_circuit_from_hamiltonian(system, system.hamiltonian, system.t, q, m) )
            #Reset Layer
            c.append( self.cooling._reset_layer(self) )
        yield c * self.cooling.options["time_steps"]
    elif(self.cooling_type == "1A"):
        for n0 in range(self.m_sys.n[0]):
            for n1 in range(self.m_sys.n[1]):
                for (g, system) in [(gx, cool_x), (gy, cool_y), (gz, cool_z)]:
                    j_int = np.zeros(shape=(1,*self.m_sys.n))
                    j_int[0, n0, n1] = g/2
                    #Reset coupling constants...
                    system._set_j_int(j_int)
                    #--- and Hamiltonian
                    system._set_hamiltonian()
                    #Trotter Layer
                    c.append( system.trotter.get_trotter_circuit_from_hamiltonian(system, system.hamiltonian, system.t, q, m) )
                    #Reset Layer
                    c.append( self.cooling._reset_layer(self) )
        yield c * self.cooling.options["time_steps"]
    
def __get_Bang_Bang_parameters(self, pauli: cirq.Gate) -> List[Real]:
    """
        Sets default BangBang parameters
        
        Parameters
        ----------
        self
        pauli: cirq.Gate
            Interaction pauli component on system side
        
        Returns
        -------
        e: Real
            Energy Gap of Two-Level system
        gamma: Real
            Coupling constant between system and ancilla
        tau: Real
            Simulation time
    """
    e = self.cooling.orth_norm(self.cooling.commutator(pauli(self.m_sys.qubits[0][0]).matrix(self.cooling.flatten(self.m_sys.qubits)), self.m_sys.hamiltonian.matrix())) #Free Spin precession
    #gamma and tau coming from Fermi's golden rule argument
    gamma = 2/np.sqrt(3) * e #Interaction constants
    tau = np.pi / gamma #Simulation time
    return e, gamma, tau

def __config_system(system: object, e: Real, gamma: Real, t: Real, pauli: cirq.Gate) -> object:
    """
        Sets the cooling object corresponding to the pauli interaction
        
        Parameters
        ----------
        system: fauvqe.CoolingModel
            Initial object that will be configured and returned
        e: Real
            Energy Gap of Two-Level system
        gamma: Real
            Coupling constant between system and ancilla
        t: Real
            Simulation time
        pauli: cirq.Gate
            Interaction component on system side
        
        Returns
        -------
        system: fauvqe.CoolingModel
            Configured cooling model
    """
    system._set_h_anc(np.transpose([e/2 * np.ones((*system.m_anc.n,))], (1, 2, 0)))
    system.t = t
    system._set_j_int(gamma/2 * system.j_int / system.j_int)
    system._two_q_gates[system.nbr_2Q_sys + system.nbr_2Q_anc:system.nbr_2Q] = [lambda q1, q2: pauli(q1)*cirq.X(q2)]
    system._set_hamiltonian()
    return system

#General Functions
def _reset_layer(self) -> cirq.Moment:
    """
        Sets the Reset Layer for cooling circuits
        
        Parameters
        ----------
        self
        
        Returns
        -------
        cirq.Moment
            Reset gate on ancilla qubit
    """
    for i in range(self.m_anc.n[0]):
        for j in range(self.m_anc.n[1]):
            yield cirq.reset(self.qubits[self.m_sys.n[0]+i][j])

def commutator(A: np.array, B:np.array) -> np.array:
    """
        Commutator of A and B
        
        Parameters
        ----------
        self
        A: np.array
            Matrix 1
        B: np.array
            Matrix 2
        
        Returns
        -------
        [A, B]
    """
    return np.dot(A,B) - np.dot(B,A)

def orth_norm(A: np.array) -> Real:
    """
        Calculates the orthogonal norm of A
        
        Parameters
        ----------
        self
        A: np.array
            matrix of which orthogonal norm is calculated
        
        Returns
        -------
        ||A||_\perp
    """
    eig_val = scipy.linalg.eigvalsh(A)
    #print((eig_val[-1] - eig_val[0])/2)
    return (eig_val[-1] - eig_val[0])/2

def flatten(list_of_lists: List[List[object]]) -> List[object]: 
    """
        Flattens a list of lists of objects into a list of objects by row-major ordering
        
        Parameters
        ----------
        self
        list_of_lists: List[List[object]]
            List of Lists that should be flattened
        
        Returns
        -------
        List[object]
    """
    return list(chain(*list_of_lists))

def ptrace(A: np.array, ind: List[np.uint]) -> np.array:
    """
        Calculates partial trace of A over the indices indicated by ind
        
        Parameters
        ----------
        self
        A: np.array
            matrix which is partially traced over
        ind: List[np.uint]
            indices which are being traced 
        
        Returns
        -------
        Tr_ind(A): np.array
    """
    #number of qubits
    n = np.log2(len(A))
    assert abs(n - int(n)) < 1e-13, "Wrong matrix size. Required 2^n, Received {}".format(n)
    n = int(n)
    #Reshape into qubit indices
    temp = A.reshape(*[2 for dummy in range(2*n)])
    count = 0
    if hasattr(ind, '__len__'):
        for i in sorted(ind, reverse=True):
            #Trace over the correct axes
            temp = np.trace(temp, axis1=i-count, axis2=n+i-2*count)
            count +=1
        #Reshape back into two-index shape
        return temp.reshape(2**(n-count), 2**(n-count))
    else:
        #Reshape back into two-index shape
        return np.trace(temp, axis1=ind, axis2=n+ind).reshape(2**(n-1), 2**(n-1))

#Backup Code, if we decide to insert variable parameters into the trotter ansatz
"""
def _get_param_resolver(self, c):
    return cirq.ParamResolver({**{"c": c}})

def set_symbols(self):
    K = self.cooling.options["K"]
    assert isinstance(p, (int, np.int_)), "Error: K needs to be int, received {}".format(type(K))
    self.circuit_param = [sympy.Symbol("c")]

def _set_c_values(self, c_values):
    p = self.cooling.options["K"]
    self.circuit_param_values[0] = c_values
"""