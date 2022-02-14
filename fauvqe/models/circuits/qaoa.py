"""
This is a submodule for Ising()

This file is not exectuded, rather called within Ising() class when:
-set_circuit('qaoa') is called 

or functions are handed over to classical optimiser
"""
import numpy as np
import cirq
import sympy
import itertools


def set_p(self, p):
    if self.qaoa.options["p"] != p:
        self.qaoa.options["p"] = p
        self.qaoa.options["append"] = False
        # Need to reset circuit
        self.qaoa.set_symbols(self)
        self.qaoa.set_circuit(self)
        # Erase beta_values, gamma_values as these are no fiting anymore
        # rename to general circuit_param_values
        # self.circuit_param_values is created in  self.qaoa.set_symbols(self, p)
        # if it does not exist
        
        #In general, this does not make sense anymore as we only 
        del self.circuit_param_values


def set_symbols(self):
    # Creat beta/gamma symbols for parametrised circuit
    # self.betas = [sympy.Symbol("b" + str(i)) for i in range(p)]
    # self.gammas = [sympy.Symbol("g" + str(i)) for i in range(p)]
    #
    # WANT: circuit-param to be a list like [b0, g0, b1, g1 ] etc
    p = self.qaoa.options["p"]
    assert isinstance(p, (int, np.int_)), "Error: p needs to be int, received {}".format(type(p))
    temp = []
    for i in range(p):
        temp.append(sympy.Symbol("b" + str(i)))
        temp.append(sympy.Symbol("g" + str(i)))
    
    if not self.qaoa.options['append']:
        self.circuit_param = temp
    else:
        self.qaoa.options['i0'] = np.size(self.circuit_param)
        try:
            self.circuit_param.extend(temp)
        except:
            self.circuit_param = temp


def _set_beta_values(self, beta_values):
    p = self.qaoa.options["p"]
    i0 = self.qaoa.options['i0']

    assert np.size(beta_values) == p, "Error: size(beta_values) !=  p; {} != {}".format(
            np.size(beta_values), p
        )

    assert np.size(self.circuit_param_values) == 2 * p +i0, " np.size(self.circuit_param_values) !=  2p + i0; {} != {}".format(
            np.size(self.circuit_param_values) , 2 * p +i0
        )

    if p > 1:
        for i in range(p):
            self.circuit_param_values[i0 + 2 * i] = beta_values[i]
    else:
        self.circuit_param_values[i0 + 0] = beta_values


def _set_gamma_values(self, gamma_values):
    p = self.qaoa.options["p"]
    i0 = self.qaoa.options['i0']

    assert np.size(gamma_values) == p, "Error: size(gamma values) !=  p; {} != {}".format(
            np.size(gamma_values), p
        )

    assert np.size(self.circuit_param_values) == 2 * p +i0, " np.size(self.circuit_param_values) !=  2p + i0; {} != {}".format(
            np.size(self.circuit_param_values) , 2 * p +i0
        )

    if p > 1:
        for i in range(p):
            self.circuit_param_values[i0 + 2 * i + 1] = gamma_values[i]
    else:
        self.circuit_param_values[i0 + 1] = gamma_values


def set_circuit(self):
    p = self.qaoa.options["p"] 
    # Reset circuit
    if not self.qaoa.options['append']:
        self.circuit = cirq.Circuit()
        i0 = 0
        self.circuit_param_values = np.zeros(2 * p)
    else:
        i0 = np.size(self.circuit_param_values)
        self.circuit_param_values = np.append(self.circuit_param_values, np.zeros(2 * p))
        
    self.qaoa.options['i0'] = i0

    #Append gates to self.circuit
    if self.qaoa.options["H_layer"]:
        self.circuit.append(self.basics._hadamard_layer(self))

    for i in range(p):
        self.circuit.append(cirq.Moment(self.qaoa._UB_layer(self, self.circuit_param[i0 + 2 * i])))
        self.circuit.append(self.qaoa._UC_layer(self, self.circuit_param[i0 + 2 * i + 1]))
        if self.field == "Z":
            self.circuit.append(cirq.Moment(self.qaoa._Z_layer(self, self.circuit_param[i0 + 2 * i + 1])))


def _UB_layer(self, beta):
    """Generator for U(beta, B) layer (mixing layer) of QAOA"""
    for row in self.qubits:
        for qubit in row:
            yield cirq.X(qubit) ** beta


def _UC_layer(self, gamma):
    """
    Generator for U(gamma, C) layer of QAOA

    Args:
      gamma: Float variational parameter for the circuit
      self.h: Array of floats of external magnetic field values
      self.J_v, self.J_h, vertikal/horizontal Interaction Hamiltonian ZZ -h Z:
      U(\gamma, C) = \prod_{\langle i,j\rangle}e^{-i\pi\gamma Z_iZ_j/2} \prod_i e^{-i\pi \gamma h_i Z_i/2
    
    for i in range(self.n[0]):
        for j in range(self.n[1]):
            if i < self.n[0] - 1:
                yield cirq.ZZ(self.qubits[i][j], self.qubits[i + 1][j]) ** (gamma * self.j_v[i, j])

            if j < self.n[1] - 1:
                yield cirq.ZZ(self.qubits[i][j], self.qubits[i][j + 1]) ** (gamma * self.j_h[i, j])

            if self.field == "Z":
                yield cirq.Z(self.qubits[i][j]) ** (gamma * self.h[i, j])
            # This X gate does not make a difference as UB layer are XpowerGates
            #elif self.field =="X":
            #    yield cirq.X(self.qubits[i][j]) ** (gamma * self.h[i, j])
    """
    for k in range(2):
        if self.n[0] > 1:
            for j in np.arange(0, self.n[1]-1+0.1, 1, dtype=int):
                #Bulk terms
                for i in np.arange(int(k), self.n[0]-1, 2, dtype=int):
                    yield cirq.ZZ(self.qubits[i][j], self.qubits[i+1][j]) ** (gamma * self.j_v[i, j])
                #Boundary terms
                if self.boundaries[0] == 0 and self.n[0]%2 == int(1-k):
                    yield cirq.ZZ(self.qubits[self.n[0]-1][j], self.qubits[0][j])** (gamma * self.j_v[self.n[0]-1, j])

        if self.n[1] > 1:
            for i in np.arange(0, self.n[0]-1+0.1, 1, dtype=int):
                #Bulk terms
                for j in np.arange(k, self.n[1]-1, 2, dtype=int):
                    yield  cirq.ZZ(self.qubits[i][j], self.qubits[i][j+1])** (gamma * self.j_h[i, j])
                #Boundary terms
                if self.boundaries[1] == 0 and self.n[1]%2 == int(1-k):
                    yield cirq.ZZ(self.qubits[i][self.n[1]-1], self.qubits[i][0])** (gamma * self.j_h[i, self.n[1]-1])

def _Z_layer(self, gamma):
    for i in range(self.n[0]):
        for j in range(self.n[1]):
            yield cirq.Z(self.qubits[i][j]) ** (gamma * self.h[i, j])

def _get_param_resolver(self, beta_values, gamma_values):
    p = self.qaoa.options["p"] 
    assert p == np.size(beta_values), "Error: p = np.size(beta_values) required"
    assert p == np.size(
        gamma_values
    ), "Error: p = np.size((self.circuit_param[1]) == len(gamma_values) required"
    # order does not mater for python dictonary
    if p == 1:
        joined_dict = {**{"b0": beta_values}, **{"g0": gamma_values}}
        # This is the same as:
        # joined_dict = {**{"b" + str(i): beta_values for i in range(p)},\
        #              **{"g" + str(i): gamma_values for i in range(p)}}
    else:
        joined_dict = {
            **{"b" + str(i): beta_values[i] for i in range(p)},
            **{"g" + str(i): gamma_values[i] for i in range(p)},
        }
    # Need return here, as yield does not work.
    return cirq.ParamResolver(joined_dict)