"""
    Include here methods that are relevant for different circuit ansätze
    To avoid replicates


    self.circuit.moments[1]._operations[0]._gate.__dict__
"""
# external import
import numpy as np
import cirq
import sympy

def rm_unused_cpv(self):
    print(self.circuit_param)
    _erase_vec = self.circuit_param.copy()
    
    #1. Find all circuit parameter|variables that are used
    for moment in self.circuit.moments:
        #print(moment.__dict__)
        for operation in moment._operations:
            print(operation._gate.__dict__.values())
            symbols = list(operation._gate.__dict__.values())
            for element in symbols:
                try:
                    print(element.atoms(sympy.Symbol))
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

def set_circuit(self):
    if not self.basics.options['append']:
        self.circuit = cirq.Circuit()

    if self.basics.options["start"] == "neel":
        self.circuit.insert(0,self.basics._neel_layer(self))
    elif self.basics.options["start"] == "mf":

    else:
        assert (False), "Invalid hea parametrisation option, received: '{}', allowed is \n \
                'joint', 'layerwise' and 'individual'".format(self.hea.options['parametrisation'] )