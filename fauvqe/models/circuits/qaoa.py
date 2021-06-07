"""
This is a submodule for Ising()

This file is not exectuded, rather called within Ising() class when:
-set_circuit('qaoa') is called 

or functions are handed over to classical optimiser
"""
import numpy as np
import cirq
import sympy


def set_p(self, p, options: dict = {"append": False}):
    try:
        if self.p != p:
            # Need to reset circuit
            self.qaoa.set_symbols(self, p)
            self.qaoa.set_circuit(self, options)
            # Erase beta_values, gamma_values as these are no fiting anymore
            # rename to general circuit_param_values
            # self.circuit_param_values is created in  self.qaoa.set_symbols(self, p)
            # if it does not exist
            del self.circuit_param_values
    except AttributeError:
        # if self.p not even exists
        self.qaoa.set_symbols(self, p)
        self.qaoa.set_circuit(self, options)
        del self.circuit_param_values


def set_symbols(self, p):
    # Creat beta/gamma symbols for parametrised circuit
    # self.betas = [sympy.Symbol("b" + str(i)) for i in range(p)]
    # self.gammas = [sympy.Symbol("g" + str(i)) for i in range(p)]
    #
    # WANT: circuit-param to be a list like [b0, g0, b1, g1 ] etc
    assert isinstance(p, (int, np.int_)), "Error: p needs to be int, received {}".format(type(p))
    temp = []
    for i in range(p):
        temp.append(sympy.Symbol("b" + str(i)))
        temp.append(sympy.Symbol("g" + str(i)))
    
    if not self.qaoa.options['append']:
        self.circuit_param = temp
    else:
        self.circuit_param.extend(temp)
    self.p = p


def _set_beta_values(self, beta_values):
    try:
        assert np.size(beta_values) == self.p, "Error: size(beta_values !=  p; {} != {}".format(
            np.size(beta_values), self.p
        )
    except AttributeError:
        # set p to length beta_values if it does not exist
        print(
            "Set self.p to np.size(beta_values) = {}, as not defined".format(np.size(beta_values))
        )
        self.p = np.size(beta_values)

    # catch if self. circuit_param_values does not exist yet
    # also require that circuit_param_values = 2*p this conflicts with the append option, resolve later if needed
    try:
        if np.size(self.circuit_param_values) != 2 * self.p:
            self.circuit_param_values = np.zeros(2 * self.p)
    except AttributeError:  # self.circuit_param_values has not been initialised earlier
        self.circuit_param_values = np.zeros(2 * self.p)

    if self.p > 1:
        for i in range(self.p):
            self.circuit_param_values[2 * i] = beta_values[i]
    else:
        self.circuit_param_values[0] = beta_values


def _set_gamma_values(self, gamma_values):
    try:
        assert np.size(gamma_values) == self.p, "Error: size(gamma values) !=  p; {} != {}".format(
            np.size(gamma_values), self.p
        )
    except AttributeError:
        # set p to length beta_values if it does not exist
        print(
            "Set self.p to np.size(gamma_values) = {}, as not defined".format(np.size(gamma_values))
        )
        self.p = np.size(gamma_values)

    # catch if self. circuit_param_values does not exist yet
    # also require that circuit_param_values = 2*p this conflicts with the append option, resolve later if needed
    try:
        if np.size(self.circuit_param_values) != 2 * self.p:
            self.circuit_param_values = np.zeros(2 * self.p)
    except AttributeError:  # self.circuit_param_values has not been initialised earlier
        self.circuit_param_values = np.zeros(2 * self.p)

    if self.p > 1:
        for i in range(self.p):
            self.circuit_param_values[2 * i + 1] = gamma_values[i]
    else:
        self.circuit_param_values[1] = gamma_values


def set_circuit(self, options):
    # Reset circuit
    if not options['append']:
        self.circuit = cirq.Circuit()

    # catch if self. circuit_param_values does not exist yet
    # also require that circuit_param_values = 2*p this conflicts with the append option, resolve later if needed
    try:
        if np.size(self.circuit_param_values) < 2 * self.p:
            self.circuit_param_values = np.zeros(2 * self.p)
    except AttributeError:  # self.circuit_param_values has not been initialised earlier
        self.circuit_param_values = np.zeros(2 * self.p)

    for i in range(self.p):
        # Issue: how to call _UB_layer, _UC_layer correctly?
        # Whatch out self.circuit_param[2*i] is the former self.betas()!!!
        # Whatch out self.circuit_param[2*i+1] is the former self.gamma()!!!
        # need to use cirq.ParamResolver and simulate(), compare final_state_vector
        self.circuit.append(cirq.Moment(self.qaoa._UB_layer(self, self.circuit_param[2 * i])))
        self.circuit.append(self.qaoa._UC_layer(self, self.circuit_param[2 * i + 1]))


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
    """
    for i in range(self.n[0]):
        for j in range(self.n[1]):
            if i < self.n[0] - 1:
                yield cirq.ZZ(self.qubits[i][j], self.qubits[i + 1][j]) ** (gamma * self.j_v[i, j])

            if j < self.n[1] - 1:
                yield cirq.ZZ(self.qubits[i][j], self.qubits[i][j + 1]) ** (gamma * self.j_h[i, j])

            yield cirq.Z(self.qubits[i][j]) ** (gamma * self.h[i, j])


def _get_param_resolver(self, beta_values, gamma_values):
    try:
        self.p == self.p
    except AttributeError:
        # set p to length beta_values if it does not exist
        self.p = np.size(beta_values)
    assert self.p == np.size(beta_values), "Error: self.p = np.size(beta_values) required"
    assert self.p == np.size(
        gamma_values
    ), "Error: p = np.size((self.circuit_param[1]) == len(gamma_values) required"
    # order does not mater for python dictonary
    if self.p == 1:
        joined_dict = {**{"b0": beta_values}, **{"g0": gamma_values}}
        # This is the same as:
        # joined_dict = {**{"b" + str(i): beta_values for i in range(self.p)},\
        #              **{"g" + str(i): gamma_values for i in range(self.p)}}
    else:
        joined_dict = {
            **{"b" + str(i): beta_values[i] for i in range(self.p)},
            **{"g" + str(i): gamma_values[i] for i in range(self.p)},
        }
    # Need return here, as yield does not work.
    return cirq.ParamResolver(joined_dict)
