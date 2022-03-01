"""
This is a submodule for Cooling1A() and CoolingNA()

This file is not exectuded, rather called within Cooling classes when:
-set_circuit('cooling') is called 

or functions are handed over to classical optimiser
"""
import numpy as np
import cirq
import sympy
import itertools

def set_K(self, K):
    if self.cooling.options["K"] != K:
        self.cooling.options["K"] = K
        self.cooling.options["append"] = False
        # Need to reset circuit
        self.cooling.set_symbols(self)
        self.cooling.set_circuit(self)

def set_circuit(self):
    K = self.cooling.options["K"]
    # Reset circuit
    if not self.cooling.options['append']:
        self.circuit = cirq.Circuit()
        self.circuit_param_values = np.zeros(1)
    if(K==1):
        raise NotImplementedError()
        self.circuit.append(self.cooling.BangBangProtocol(self)) #TODO
    else:
        self.circuit.append(self.cooling.LogSweepProtocol(self))

#LogSweep
def LogSweepProtocol(self):
    K = self.cooling.options["K"]
    if(self.cooling.options["emin"] is None or self.cooling.options["emax"] is None):
        e_min, e_max, spectral_spread = self.cooling.__get_default_e_m(self)
        print("Using default values for emin: {} and emax: {}".format(e_min, e_max))
    else:
        e_min = self.cooling.options["emin"]
        e_max = self.cooling.options["emax"]
        m = self.cooling.options["m"]
    for k in range(K):
        e, g, tau = self.cooling.__get_Log_Sweep_parameters(self, e_min, e_max, k)
        m = math.ceil(2*np.sqrt(1+ (spectral_spread**2)/(g**2)))
        #TODO: Set j_int for different n_sys and set Hamiltonian newly before handing over to Trotter routine
        yield self.trotter.get_trotter_circuit_from_hamiltonian(self, self.hamiltonian, self.t, 1, m)
        for i in range(self.m_anc.n[0]):
            for j in range(self.m_anc.n[1]):
                yield cirq.reset(self.qubits[self.m_sys.n[0]+i][j])

def __get_default_e_m(self):
    _N = 2**np.size(self.qubits)
    if np.size(self.eig_val) != _N or (np.shape(self.eig_vec) != np.array((_N, _N)) ).all():
        self.diagonalise(solver = "scipy", solver_options={"subset_by_index": [0, _N - 1]})
    energy_ex = self.eig_val[0]
    energy_ex2 = self.eig_val[1]
    if(energy_ex2 - energy_ex < 1e-7):
        energy_ex2 = self.eig_val[2]
    e_min = (energy_ex2 - energy_ex)
    spectral_spread = (self.eig_val[-1] - energy_ex)
    e_max = max( [ self.cooling.__orth_norm(self.cooling.__commutator(pauli(self.qubits[0][0]).matrix(helpers.flatten(self.qubits)), self.hamiltonian.matrix())) for pauli in [cirq.X, cirq.Y, cirq.Z] ] )
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
def __get_Bang_Bang_parameters(self, interaction, system):
    e = orth_norm(commutator(interaction(system.qubits[0][0]).matrix(flatten(system.qubits)), 
                                             system.hamiltonian.matrix())) #Free Spin precession
    g = 2/np.sqrt(3) * e #Interaction constants
    if(g!=0):
        tau = np.pi / g #Simulation time
    else:
        tau = 0
    return e, g, tau

#General Functions
def __commutator(A, B):
    return A@B - B@A

def __orth_norm(A):
    mask = np.ones(A.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    return abs(A)[mask].max()

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