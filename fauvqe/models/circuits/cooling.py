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

def set_symbols(self):
    K = self.cooling.options["K"]
    assert isinstance(p, (int, np.int_)), "Error: K needs to be int, received {}".format(type(K))
    self.circuit_param = [sympy.Symbol("c")]

def _set_c_values(self, c_values):
    p = self.cooling.options["K"]
    self.circuit_param_values[0] = c_values

def set_circuit(self):
    K = self.cooling.options["K"]
    # Reset circuit
    if not self.cooling.options['append']:
        self.circuit = cirq.Circuit()
        self.circuit_param_values = np.zeros(1)
    
    for k in range(K):
        self.circuit.append(self.cooling._trotter_layer(self, self.circuit_param[0]))
        for i in range(self.m_anc.n[0]):
            for j in range(self.m_anc.n[1]):
                self.circuit.append(cirq.reset(self.qubits[self.m_sys.n[0]+i][j]))

def _get_param_resolver(self, c):
    return cirq.ParamResolver({**{"c": c}})