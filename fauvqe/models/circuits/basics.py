"""
    Include here methods that are relevant for different circuit ansätze
    To avoid replicates


    self.circuit.moments[1]._operations[0]._gate.__dict__
"""
# external import
import numpy as np
import cirq
import sympy

def _hadamard_layer(self):
    for row in self.qubits:
        for qubit in row:
            yield cirq.H.on(qubit)

def rm_unused_cpv(self):
    #print(self.circuit_param)
    _erase_vec = self.circuit_param.copy()
    
    #1. Find all circuit parameter|variables that are used
    for moment in self.circuit.moments:
        #print(moment.__dict__)
        for operation in moment._operations:
            #print(operation._gate.__dict__.values())
            symbols = list(operation._gate.__dict__.values())
            for element in symbols:
                try:
                    #print(element.atoms(sympy.Symbol))
                    symbol = element.atoms(sympy.Symbol)
                    _erase_vec=set(_erase_vec) - symbol
                except:
                    pass 
    
    #2. Erase elements that are still in _erase_vec from self.circuit_param
    _erase_ind = []
    for redundant_element in _erase_vec:
        _erase_ind.append(self.circuit_param.index(redundant_element))
    
    _erase_ind.sort(reverse = True)
    for index in _erase_ind:
        del self.circuit_param[index]
    self.circuit_param_values = np.delete(self.circuit_param_values, _erase_ind, None)

def _neel_layer(self):
    for row in self.qubits:
        for qubit in row:
            # Possibly only specific to Grid Qubit:
            if (qubit._row + qubit._col)%2 == 0:
                yield cirq.X(qubit)


def _mf_layer(self):
    for row in self.qubits:
        for qubit in row:
            theta = self.basics.__get_mf_angle(self, qubit)
            #print("theta: \t{}".format(theta))
            yield cirq.XPowGate(exponent=(theta/np.pi)).on(qubit)

def __get_mf_angle(self, qubit):
    #1. Calculate mean J
    #        (1)
    #         |
    #   (2) - x - (3)
    #         |
    #        (4)
    #
    J_mean = 0
    n_J = 0
    
    if (self.j_v.shape[0] * self.j_v.shape[1]) > 0:
        # (1)
        if qubit._row > 0:
            J_mean += self.j_v[qubit._row - 1,  qubit._col]
            n_J += 1
        
        # (4)
        if qubit._col <  self.j_v.shape[1] and qubit._row <  self.j_v.shape[0]:
            J_mean += self.j_v[qubit._row, qubit._col]
            n_J += 1
    
    if (self.j_h.shape[0] * self.j_h.shape[1]) > 0: 
        # (2)
        if qubit._col > 0:
            J_mean += self.j_h[qubit._row, qubit._col -1]
            n_J += 1

        # (3)
        if qubit._col <  self.j_h.shape[1] and qubit._row <  self.j_h.shape[0]:
            J_mean += self.j_h[qubit._row, qubit._col]
            n_J += 1

    # Get and return theta via
    #sin theta = h_x/J_mean for spins
    # need to map to qubit
    # Hence use theta = (-1)^(col+row)*arccos(1- (h_ij/J(0))^2)^0.5
    J_mean /= n_J
    #print("_col: \t{}, _row\t{}".format(qubit._col , qubit._row))
    #print("n_J: \t{}, \nJ_mean \t {}".format(n_J, J_mean))
    #print("h_ij: \t{}".format(self.h[qubit._row, qubit._col]))
    #print("self.j_v.shape \t {}".format(self.j_v.shape))
    #print("self.j_h.shape \t {}".format(self.j_h.shape))

    if abs(self.h[qubit._row, qubit._col]) > abs(J_mean):
        return np.pi/2
    else:
        return (-1)**(qubit._col+qubit._row)*\
            np.arccos(np.sqrt(1-(self.h[qubit._row, qubit._col]/J_mean)**2))

def set_circuit(self):
    if not self.basics.options['append']:
        self.circuit = cirq.Circuit()

    if self.basics.options["start"] == "hadamard":
        self.circuit.insert(0,self.basics._hadamard_layer(self))
    elif self.basics.options["start"] == "neel":
        self.circuit.insert(0,self.basics._neel_layer(self))
    elif self.basics.options["start"] == "mf":
        self.circuit.insert(0,self.basics._mf_layer(self))
    else:
        assert (False), "Invalid self.basics option, received: '{}', allowed is \n \
                'hadamard', 'neel',  and 'mf'".format(self.hea.options['parametrisation'] )