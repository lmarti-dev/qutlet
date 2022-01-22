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
from numbers import Number


def _1Qubit_layer(self, i, g):
    """
        Generator for general 1Qubit layer

        a, x, z are possibly defined in 
        if not set 0 
    """
    
    gate = self.hea.options["1QubitGates"][g]
    if self.hea.options["parametrisation"] in {"joint", "layerwise"}:
        variables = [0 for dummy in self.hea.options["1Qvariables"][g]]
        j = 0
        for variable in self.hea.options["1Qvariables"][g]:
            variables[j] = sympy.Symbol(variable + str(i))
            j +=1
        
        for row in self.qubits:
            for qubit in row:
                yield gate(*variables).on(qubit)
    elif self.hea.options["parametrisation"] == "individual":
        v_mask = [0 for dummy in self.hea.options["1Qvariables"][g]]
        j = 0
        for variable in self.hea.options["1Qvariables"][g]:
            v_mask[j] = variable
            j +=1
        
        #print(v_mask)
        
        j = 0
        #sympy.Symbol(variable + str(i) + "_" + str(j))
        for row in self.qubits:
            for qubit in row:
                variables = [0 for dummy in self.hea.options["1Qvariables"][g]]
                for k in range(len(self.hea.options["1Qvariables"][g])):
                    variables[k] = self.hea.__get_sympy_Symbol(self, v_mask[k],i,j)
                yield gate(*variables).on(qubit)
                #print(variables)
                j +=1
    else:
        assert (False), "Invalid hea parametrisation option, received: '{}', allowed is \n \
            'joint', 'layerwise' and 'individual'".format(self.hea.options['parametrisation'] )


def _partial_2Qubit_layer(self, i_p, v_v, v_h, g):
    """
    Generator for hardware efficent 2 qubit layer

    Args:
      i layer number
      i_p partial layer number
      self.hea.options["2quibtgate"]
      
      variables array of sympy symbols that parametrieses circuit
    """
    gate = self.hea.options["2QubitGates"][g]
    
    # if i_p 0 or 2
    #less specific: i_p%2 == 0
    if i_p in [0,2]:
        #Need to be smarter ordered!!!
        if self.n[0] > 1:
            for j in np.arange(0, self.n[1]-1+0.1, 1, dtype=int):
                #Bulk terms
                for i in np.arange(int(i_p/2), self.n[0]-1, 2, dtype=int):
                    #print("i: \t{}, j: \t{}".format(i,j))
                    #print(*v_v[:][i][j])
                    yield gate(*v_v[:][i][j]).on(self.qubits[i][j], self.qubits[i+1][j])
                #Boundary terms
                if self.boundaries[0] == 0 and self.n[0]%2 == int(1-i_p/2):
                    yield gate(*v_v[:][self.n[0]-1][j]).on(self.qubits[self.n[0]-1][j], self.qubits[0][j])

    # if i_p 1 or 3
    #less specific: i_p%2 == 1
    elif i_p in [1,3]:
        #Need to be smarter ordered!!!
        if self.n[1] > 1:
            for i in np.arange(0, self.n[0]-1+0.1, 1, dtype=int):
                #Bulk terms
                for j in np.arange(int((i_p-1)/2), self.n[1]-1, 2, dtype=int):
                    #print("i: \t{}, j: \t{}".format(i,j))
                    yield gate(*v_h[:][i][j]).on(self.qubits[i][j], self.qubits[i][j+1])
                #Boundary terms
                if self.boundaries[1] == 0 and self.n[1]%2 == int(1-(i_p-1)/2):
                    yield gate(*v_h[:][:][i][self.n[1]-1]).on(self.qubits[i][self.n[1]-1], self.qubits[i][0])
            
def _2Qubit_layer(self, i, g):
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
        gate_variables = [0 for dummy in self.hea.options["2Qvariables"][g]]
        j = 0
        for variable in self.hea.options["2Qvariables"][g]:
            gate_variables[j] = sympy.Symbol(variable + str(i))
            j +=1
        
        temp = [gate_variables for i in range(self.n[1])]
        v_v = [temp for i in range(self.n[0]-self.boundaries[0])]
        
        temp = [gate_variables for i in range(self.n[1]-self.boundaries[1])]
        v_h = [temp for i in range(self.n[0])]

        for i_p in range(4):
            yield self.hea._partial_2Qubit_layer(self, i_p, v_v, v_h, g)

    elif self.hea.options["parametrisation"] == "layerwise":
        for i_p in range(4):
            gate_variables = [0 for dummy in self.hea.options["2Qvariables"][g]]
            j = 0
            for variable in self.hea.options["2Qvariables"][g]:
                gate_variables[j] = sympy.Symbol(variable + str(i) + "_" + str(i_p))
                j +=1
            
            temp = [gate_variables for i in range(self.n[1])]
            v_v = [temp for i in range(self.n[0]-self.boundaries[0])]
        
            temp = [gate_variables for i in range(self.n[1]-self.boundaries[1])]
            v_h = [temp for i in range(self.n[0])]

            yield self.hea._partial_2Qubit_layer(self, i_p, v_v, v_h, g)

    elif self.hea.options["parametrisation"] == "individual":
        for i_p in range(4):
            gate_variables = [0 for dummy in self.hea.options["2Qvariables"][g]]
            j = 0
            for variable in self.hea.options["2Qvariables"][g]:
                gate_variables[j] = sympy.Symbol(variable + str(i) + "_" + str(i_p))
                j +=1
            
            v_mask = [0 for dummy in self.hea.options["2Qvariables"][g]]
            h_mask = [0 for dummy in self.hea.options["2Qvariables"][g]]
            j = 0
            for variable in self.hea.options["2Qvariables"][g]:
                v_mask[j] = variable + str(i) + "_" + str(i_p) + "_v"
                h_mask[j] = variable + str(i) + "_" + str(i_p) + "_h"
                j +=1
            
            v_v = []
            for l in range(self.n[0]-self.boundaries[0]):
                temp = []
                for j in range(self.n[1]):
                    temp.append([self.hea.__get_sympy_Symbol(self, v_mask[0],l,j),
                                 self.hea.__get_sympy_Symbol(self, v_mask[1],l,j)])
                #print("temp:\n {}".format(temp))
                v_v.append(temp)
            #print("v_v:\n {}".format(v_v))

            v_h = []
            for l in range(self.n[0]):
                temp = []
                for j in range(self.n[1]-self.boundaries[1]):
                    temp.append([self.hea.__get_sympy_Symbol(self, h_mask[0],l,j),
                                 self.hea.__get_sympy_Symbol(self, h_mask[1],l,j)])
                #print("temp:\n {}".format(temp))
                v_h.append(temp)
            #print("v_h:\n {}".format(v_h))  
            #temp = [gate_variables for i in range(self.n[1]-self.boundaries[1])]
            #v_h = [temp for i in range(self.n[0])]

            yield self.hea._partial_2Qubit_layer(self, i_p, v_v, v_h, g)
    else:
        assert (False), "Invalid hea parametrisation option, received: '{}', allowed is \n \
            'joint', 'layerwise' and 'individual'".format(self.hea.options['parametrisation'] )

def __get_sympy_Symbol(self, v_mask_element,i,j):
    if isinstance(v_mask_element, str):
        return sympy.Symbol(v_mask_element + str(i) + "_" + str(j))
    elif isinstance(v_mask_element, Number):
        return v_mask_element
    #else:
    #    assert (False), "In HEA: invalid input type in __get_sympy_Symbol() received: '{}', allowed is \n \
    #        'str' or any number type".format(type(v_mask_element))

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
    temp1Q = []
    temp2Q = []
    
    for i in range(self.hea.options["p"]):
        if self.hea.options['parametrisation'] == 'joint':
            #enforce alphabetic order
            for variable in sorted(sum(self.hea.options["1Qvariables"], [])):
                temp1Q.append(sympy.Symbol(variable + str(i)))
                
            for variable in sorted(sum(self.hea.options["2Qvariables"], [])):
                temp2Q.append(sympy.Symbol(variable + str(i)))
        
        elif self.hea.options['parametrisation'] == 'layerwise':
            for variable in sorted(sum(self.hea.options["1Qvariables"], [])):
                #1qubit cases
                temp1Q.append(sympy.Symbol(variable + str(i)))
                
            for variable in sorted(sum(self.hea.options["2Qvariables"], [])):
                #2qubit cases
                for j in range(4):
                    temp2Q.append(sympy.Symbol(variable + str(i) + "_" + str(j)))

        elif self.hea.options['parametrisation'] == 'individual':
            #enforce alphabetic order
            for variable in sorted(sum(self.hea.options["1Qvariables"], [])):
                #1qubit cases
                for j in range(np.size(self.qubits)):
                    temp1Q.append(sympy.Symbol(variable + str(i) + "_" + str(j)))
            
            for variable in sorted(sum(self.hea.options["2Qvariables"], [])):
                #2qubit cases
                #Here: only count number of parameters. 
                for j in range(np.size(self.j_v[0]) + np.size(self.j_h[0])):
                    temp2Q.append(sympy.Symbol(variable + str(i) + "_" + str(j)))
        else:
            assert (False), "Invalid hea parametrisation option, received: '{}', allowed is \n \
                'joint', 'layerwise' and 'individual'".format(self.hea.options['parametrisation'] )
    if not self.hea.options['append'] or self.circuit_param is None:
        self.circuit_param = temp1Q + temp2Q
    else:
        self.circuit_param.extend(temp1Q + temp2Q)
    self.p = self.hea.options["p"]

def set_circuit(self):
    # Reset circuit
    if not self.hea.options['append']:
        self.circuit = cirq.Circuit()

    if np.size(self.circuit_param_values) != np.size(self.circuit_param):
        self.circuit_param_values = np.zeros(np.size(self.circuit_param))

    for i in range(self.p):
        if self.hea.options["1QubitGates"] is not None:
            for g in range(len(self.hea.options["1QubitGates"])):
                self.circuit.append(cirq.Moment(self.hea._1Qubit_layer(self, i, g)))
                #self.circuit.append(cirq.Moment(self.hea._PhXZ_layer(self, i)))
        if self.hea.options["2QubitGates"] is not None:
            for g in range(len(self.hea.options["2QubitGates"])):
                self.circuit.append(self.hea._2Qubit_layer(self, i, g))
    #Erase hea circuit parameters, that are not used
    # e.g. in 1D case
    # Can should be more general/basic function

def set_circuit_param_values(self, key, values):
    """
        Receives one or more parameter keys
            e.g. "a", "x", "z", "phi", "theta", "b", "g" 

        Sets/resets all self.circuit_param_values that fit to respective parameter key

        1. Find all self.circuit_param that fit and get their index
        2. Set the respective self.cricuit_param
    """
    indices = []
    temp_variables = []
    for i in range(len(self.circuit_param)):
        if str(self.circuit_param[i]).startswith(key):
            indices.append(i)
            temp_variables.append(str(self.circuit_param[i]))
   
    assert(np.size(indices) == np.size(values)),\
        "HEAError: Number of variables ({}) != given variable values ({})".format(np.size(indices), np.size(self.circuit_param))

     #Order by alphabet/indices
    x = np.argsort(temp_variables)
    indices = np.array(indices)
    sorted_indices = indices[x]

    #Reset self.circuit_param_values
    #This also works for single parameters
    self.circuit_param_values[sorted_indices] = values
    
"""
#Not required anymore:
def _PhXZ_layer(self, i):
    ""
        Generator for PhasedXZGate Layer

        a, x, z are possibly defined in 
        if not set 0 
    ""
    if self.hea.options["parametrisation"] in {"joint", "layerwise"}:
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
    elif self.hea.options["parametrisation"] == "individual":
        v_mask = [0, 0, 0]
        j = 0
        for variable in ["a", "x", "z"]:
            if variable in self.hea.options["variables"]:
                v_mask[j] = variable
            j +=1
        j = 0
        #print(v_mask)

        #sympy.Symbol(variable + str(i) + "_" + str(j))
        for row in self.qubits:
            for qubit in row:
                yield cirq.PhasedXZGate(x_exponent=self.hea.__get_sympy_Symbol(self, v_mask[1],i,j), 
                                        z_exponent=self.hea.__get_sympy_Symbol(self, v_mask[2],i,j), 
                                        axis_phase_exponent=self.hea.__get_sympy_Symbol(self, v_mask[0],i,j)).on(qubit)
                j +=1
    else:
        assert (False), "Invalid hea parametrisation option, received: '{}', allowed is \n \
            'joint', 'layerwise' and 'individual'".format(self.hea.options['parametrisation'] )

"""