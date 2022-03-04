"""
This is a submodule for Cooling1A() and CoolingNA()

This file is not exectuded, rather called within Cooling classes when:
-set_circuit('cooling') is called 

or functions are handed over to classical optimiser
"""
import numpy as np
import cirq
import sympy
import scipy
import math
from itertools import chain

import fauvqe 

def set_K(self, K):
    if self.cooling.options["K"] != K:
        self.cooling.options["K"] = K
        self.cooling.options["append"] = False
        # Need to reset circuit
        self.cooling.set_circuit(self)

def set_circuit(self):
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
def LogSweepProtocol(self):
    K = self.cooling.options["K"]
    m = self.cooling.options["m"]
    q = self.cooling.options["q"]
    if(self.cooling.options["emin"] is None or self.cooling.options["emax"] is None):
        e_min, e_max, spectral_spread = self.cooling.__get_default_e_m(self)
        print("Using default values for emin: {} and emax: {} and spectral spread: {}".format(e_min, e_max, spectral_spread))
    else:
        e_min = self.cooling.options["emin"]
        e_max = self.cooling.options["emax"]
    for k in range(K):
        e, g, tau = self.cooling.__get_Log_Sweep_parameters(self, e_min, e_max, k)
        if(self.cooling.options["m"] is None):
            m = math.ceil(2*np.sqrt(1+ (spectral_spread**2)/(g**2)))
        self._set_h_anc(np.transpose([e/2 * np.ones((*self.m_anc.n,))], (1, 2, 0)))
        self.t = tau
        c = cirq.Circuit()
        if(isinstance(self, fauvqe.CoolingNA)):
            self._set_j_int(g/2 * self.j_int / self.j_int)
            self._set_hamiltonian()
            c.append( self.trotter.get_trotter_circuit_from_hamiltonian(self, self.hamiltonian, self.t, q, m) )
            c.append( self.cooling._reset_layer(self) )
            yield c * self.cooling.options["time_steps"]
        elif(isinstance(self, fauvqe.Cooling1A)):
            for n0 in range(self.m_sys.n[0]):
                for n1 in range(self.m_sys.n[1]):
                    j_int = np.zeros(shape=(1,*self.m_sys.n))
                    j_int[0, n0, n1] = g/2
                    self._set_j_int(j_int)
                    for pauli in [cirq.X, cirq.Y, cirq.Z]:
                        self._two_q_gates[self.nbr_2Q_sys + self.nbr_2Q_anc:self.nbr_2Q] = [lambda q1, q2: pauli(q1)*cirq.X(q2)]
                        self._set_hamiltonian()
                        c.append( self.trotter.get_trotter_circuit_from_hamiltonian(self, self.hamiltonian, self.t, q, m) )
                        c.append( self.cooling._reset_layer(self) )
            yield c * self.cooling.options["time_steps"]
        
def __get_default_e_m(self):
    _n = np.size(self.m_sys.qubits)
    _N = 2**_n
    if np.size(self.m_sys.eig_val) != _N or (np.shape(self.m_sys.eig_vec) != np.array((_N, _N)) ).all():
        self.m_sys.diagonalise(solver = "scipy", solver_options={"subset_by_index": [0, _N - 1]})
    energy_ex = self.m_sys.eig_val[0]
    energy_ex2 = self.m_sys.eig_val[1]
    e_min = (energy_ex2 - energy_ex)
    spectral_spread = (self.m_sys.eig_val[-1] - energy_ex)
    e_max = max( [ self.cooling.orth_norm(self.cooling.commutator(pauli(self.m_sys.qubits[0][0]).matrix(self.cooling.flatten(self.m_sys.qubits)), self.m_sys.hamiltonian.matrix())) for pauli in [cirq.X, cirq.Y, cirq.Z] ] )
    return e_min, e_max, spectral_spread

def __get_Log_Sweep_parameters(self, e_min, e_max, k):
    K = self.cooling.options["K"]
    e = e_min**(k/(K-1)) * e_max**(1 - k/(K-1)) #Free Spin precession
    h = e_max / e_min
    R = np.log(h) * ((1 - h) / (2 * (1 + h)) + np.log(2 * h / (1 + h)))
    delta_factor = np.log(K * 8 / R) / 2 / np.pi
    delta = delta_factor * e * (1 - 2 / (1 + (e_min / e_max)**(1 / (1 - K))))
    tau = 1/delta
    g = np.pi / tau
    return e, g, tau

#BangBang
def BangBangProtocol(self):
    if(self.cooling.options["m"] is None):
        m=1
    else:
        m = self.cooling.options["m"]
    q = self.cooling.options["q"]
    ex, gx, tx = self.cooling.__get_Bang_Bang_parameters(self, cirq.X)
    ey, gy, ty = self.cooling.__get_Bang_Bang_parameters(self, cirq.Y)
    ez, gz, tz = self.cooling.__get_Bang_Bang_parameters(self, cirq.Z)
    cool_x = self.cooling.__config_system(self.copy(), ex, gx, tx, cirq.X)
    cool_y = self.cooling.__config_system(self.copy(), ey, gy, ty, cirq.Y)
    cool_z = self.cooling.__config_system(self.copy(), ez, gz, tz, cirq.Z)
    c = cirq.Circuit()
    if(isinstance(self, fauvqe.CoolingNA)):
        for sys in [cool_x, cool_y, cool_z]:
            c.append( sys.trotter.get_trotter_circuit_from_hamiltonian(sys, sys.hamiltonian, sys.t, q, m) )
            c.append( self.cooling._reset_layer(self) )
        yield c * self.cooling.options["time_steps"]
    elif(isinstance(self, fauvqe.Cooling1A)):
        for n0 in range(self.m_sys.n[0]):
            for n1 in range(self.m_sys.n[1]):
                for (g, sys) in [(gx, cool_x), (gy, cool_y), (gz, cool_z)]:
                    j_int = np.zeros(shape=(1,*self.m_sys.n))
                    j_int[0, n0, n1] = g/2
                    sys._set_j_int(j_int)
                    sys._set_hamiltonian()
                    c.append( sys.trotter.get_trotter_circuit_from_hamiltonian(sys, sys.hamiltonian, sys.t, q, m) )
                    c.append( self.cooling._reset_layer(self) )
        yield c * self.cooling.options["time_steps"]
    
def __get_Bang_Bang_parameters(self, pauli):
    e = self.cooling.orth_norm(self.cooling.commutator(pauli(self.m_sys.qubits[0][0]).matrix(self.cooling.flatten(self.m_sys.qubits)), self.m_sys.hamiltonian.matrix())) #Free Spin precession
    g = 2/np.sqrt(3) * e #Interaction constants
    tau = np.pi / g #Simulation time
    return e, g, tau

def __config_system(system, e, g, t, pauli):
    system._set_h_anc(np.transpose([e/2 * np.ones((*system.m_anc.n,))], (1, 2, 0)))
    system.t = t
    system._set_j_int(g/2 * system.j_int / system.j_int)
    system._two_q_gates[system.nbr_2Q_sys + system.nbr_2Q_anc:system.nbr_2Q] = [lambda q1, q2: pauli(q1)*cirq.X(q2)]
    system._set_hamiltonian()
    return system

#General Functions
def _reset_layer(self):
    for i in range(self.m_anc.n[0]):
        for j in range(self.m_anc.n[1]):
            yield cirq.reset(self.qubits[self.m_sys.n[0]+i][j])

def commutator(A, B):
    return A@B - B@A

def orth_norm(A):
    eig_val = scipy.linalg.eigvalsh(A)
    return (eig_val[-1] - eig_val[0])/2

def flatten(list_of_lists): 
    return list(chain(*list_of_lists))

def ptrace(A, ind):
    n = np.log2(len(A))
    assert abs(n - int(n)) < 1e-13, "Wrong matrix size. Required 2^n, Received {}".format(n)
    n = int(n)
    temp = A.reshape(*[2 for dummy in range(2*n)])
    count = 0
    if hasattr(ind, '__len__'):
        for i in sorted(ind, reverse=True):
            temp = np.trace(temp, axis1=i-count, axis2=n+i-2*count)
            count +=1
        return temp.reshape(2**(n-count), 2**(n-count))
    else:
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