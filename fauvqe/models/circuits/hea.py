"""
This is a submodule for Ising()

This file is not exectuded, rather called within Ising() class when:
-set_circuit('hea') is called 

Hea = hardware efficent ansatz

Idea:
- PhXZ layer max 3 parameters per gate, name a,x. z
    https://quantumai.google/reference/python/cirq/ops/PhasedXZGate
    "Google devices support arbitrary one-qubit gates of any rotation. The full complement of these rotations can be accessed by 
    using the cirq.PhasedXZGate. More restrictive one-qubit gates, such as the Pauli gates cirq.X, cirq.Y, cirq.Z, as well as the 
    gate cirq.PhasedXPowGate can also be natively executed. One qubit rotations have a duration of 25 ns on most Google devices."
- i Swap layer max 1 parameter per gate? name i0_0, first layer than _ within layer
        Better fsim gate?  FSimGate 
        cirq.FSimGate(np.pi / 4, 0.0),
        FSimGate(θ, φ) = ISWAP**(-2θ/π) CZPowGate(exponent=-φ/π)
        "Contains all two qubit interactions that preserve excitations, up to single-qubit rotations and global phase."
- option: either in one layer either common or individual parameters 
    make dict?
    self.hea.options = {'parametrisation' : 'joint' vs 'individual' vs 'layerwise'
                        'append': Falsevs true
                        '2QubitGate' : 'fsim' vs 'iSWAP' vs. 'SYC'}
    layerwise gives 8 2qubit angles instead of 2

https://quantumai.google/cirq/google/devices
https://quantumai.google/reference/python/cirq/ops/FSimGate
https://quantumai.google/reference/cc/qsim/struct/qsim/cirq/f-sim-gate
"""
# external import
import numpy as np
import cirq
import sympy

def _PhXZ_layer(self, i):
    """
        Generator for PhasedXZGate Layer

        a, x, z are possibly defined in 
        if not set 0 
    """
    if self.hea.options["parametrisation"] == "joint":
        variables = [0, 0, 0]
        j = 0
        for variable in ["a", "x", "z"]:
            if variable in self.hea.options["variables"]:
                variables[j] = sympy.Symbol(variable + str(i))
            j +=1
        for row in self.qubits:
            for qubit in row:
                yield cirq.PhasedXZGate(x_exponent=variables[1], 
                                        z_exponent=variables[2], 
                                        axis_phase_exponent=variables[0]).on(qubit)


def _partial_2Qubit_layer(self, i_p, v_v, v_h):
    """
    Generator for hardware efficent 2 qubit layer

    Args:
      i layer number
      i_p partial layer number
      self.hea.options["2quibtgate"]
      
      variables array of sympy symbols that parametrieses circuit
    """
    gate = self.hea.options["2QubitGate"]
    
    # if i_p 0 or 2
    if i_p%2 == 0:
        #Need to be smarter ordered!!!
        for i in range(self.n[0]):
            #Bulk terms
            for j in range(int(i_p/2), self.n[1]-1, 2):
                yield gate(*v_v[:][i][j]).on(self.qubits[i][j], self.qubits[i][j + 1])
            #Boundary terms
            if self.boundaries[1] == 0 and self.n[1]%2 == int(1-i_p/2):
                yield gate(*v_v[:][i][self.n[1]-1]).on(self.qubits[i][self.n[1]-1], self.qubits[i][0])     

    # if i_p 1 or 3
    if i_p%2 == 0:
        #Need to be smarter ordered!!!
        for j in range(self.n[1]):
            #Bulk terms
            for i in range(0, self.n[0]-1, 2):
                yield gate(*v_h[:][i][j]).on(self.qubits[i][j], self.qubits[i + 1][j])
            #Boundary terms
            if self.boundaries[0] == 0 and self.n[0]%2 == 1:
                yield gate(*v_v[:][self.n[0]-1][j]).on(self.qubits[self.n[0]-1][j], self.qubits[0][j])
            
def _2Qubit_layer(self, i):
    """
    Generator for hardware efficent 2 qubit layer

    Args:
      i layer number
      self.hea.options["2quibtgate"]
      self.hea.options["parametrisation"]

    Do:
        1. Generate array variables dependent on boundary conditions
        2. Call _partial_2Qubit_layer(), hand over variables array
    """
    if self.hea.options["parametrisation"] == "joint":
        # Offer different defaults????
        gate_variables = [0, 0]
        j = 0
        for variable in ["theta", "phi"]:
            if variable in self.hea.options["variables"]:
                gate_variables[j] = sympy.Symbol(variable + str(i))
            j +=1

        temp = [gate_variables for i in range(self.n[1])]
        v_v = [temp for i in range(self.n[0]-self.boundaries[0])]
        
        temp = [gate_variables for i in range(self.n[1]-self.boundaries[1])]
        v_h = [temp for i in range(self.n[0])]

        for i_p in range(4):
            yield self.hea._partial_2Qubit_layer(self, i_p, v_v, v_h)

def set_symbols(self):
    """
        Creates a list of sympy Symbols to parametrise the to be create circuit

        Needs to know:
            - Individual or common parametrisation
                provide fal in options dict
            - 1qubit and 2qubit gate parametrisation
                - idea give dict with names
                - maximally 'a', 'x', 'z', 'theta', 'phi'

    """
    assert isinstance(self.hea.options["p"], (int, np.int_)), "Error: p needs to be int, received {}".format(type(self.hea.options["p"]))
    temp = []
    
    for i in range(self.hea.options["p"]):
        if self.hea.options['parametrisation'] == 'joint':
            #enforce alphabetic order
            for variable in sorted(self.hea.options["variables"]):
                temp.append(sympy.Symbol(variable + str(i)))
        elif self.hea.options['parametrisation'] == 'individual':
            #enforce alphabetic order
            for variable in sorted(self.hea.options["variables"]):
                #1qubit cases
                if variable in {"a", "x", "z"}:
                    for j in range(np.size(self.qubits)):
                        temp.append(sympy.Symbol(variable + str(i) + "_" + str(j)))
                #1qubit cases
                elif variable in {"phi", "theta"}:
                    #Here: only count number of parameters. 
                    for j in range(np.size(self.j_v) + np.size(self.j_h)):
                        temp.append(sympy.Symbol(variable + str(i) + "_" + str(j)))
        else:
            assert (False), "Invalid hea parametrisation option, received: '{}', allowed is \n \
                'joint' and 'individual'".format(self.hea.options['parametrisation'] )
    self.circuit_param = temp
    self.p = self.hea.options["p"]

def set_circuit(self):
    # Reset circuit
    if not self.hea.options['append']:
        self.circuit = cirq.Circuit()

    try:
        if np.size(self.circuit_param_values) != np.size(self.circuit_param):
            self.circuit_param_values = np.zeros(np.size(self.circuit_param))
    except AttributeError:  # self.circuit_param_values has not been initialised earlier
        self.circuit_param_values = np.zeros(np.size(self.circuit_param))

    for i in range(self.p):
        self.circuit.append(cirq.Moment(self.hea._PhXZ_layer(self, i)))
        self.circuit.append(self.hea._2Qubit_layer(self, i))
        #self.circuit.append(self.hea._2Qubit_layer(self, i))

def set_circuit_param_values(self):
    """
        Receives one or more parameter keys
            "a", "x", "z", "phi", "theta"

        Sets/resets all self.circuit_param_values that fit to respective parameter key
    """
    pass

"""


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


def set_circuit(self, append):
    # Reset circuit
    if not append:
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





"""