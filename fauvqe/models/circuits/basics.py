"""
    Include here methods that are relevant for different circuit ansätze
    To avoid replicates


    self.circuit.moments[1]._operations[0]._gate.__dict__
"""
from fauvqe.models.abstractmodel import AbstractModel

# external import
import numpy as np
import cirq
import sympy
import warnings
import itertools 

from typing import Literal
from numbers import Real

def _exact_layer(self):
    #1. Check whether given subgrid is multiple of overall grid
    #2. Use this to determine where/ how to apply it
    #3. Need to generate and solve smaller proxy systems
    #
    #   How to cut a circuit to sub-systems?
    #   example 2x4 to (1,2) subsystems
    #
    #   x - x - x - x -     x - x, x - x -,
    #   |   |   |   |    => x - x or x - x - 
    #   x - x - x - x -     |   |    |   |  
    #
    #   hard to tell what's best => further flag
    #Further have to include boundary condition of large system for i = n_exact[0] or j = n_exact[1]


    if self.qubittype != "GridQubit":
        raise NotImplementedError()
    
    n_exact = self.basics.options["n_exact"]
    b_exact = self.basics.options["b_exact"]
    print("self.n: \t {},n_exact \t {},b_exact \t {}".format(self.n, n_exact, b_exact))
    if np.sum(self.n%n_exact) != 0:
        warnings.warn("IsingBasicsWarning: self.n%n_exact != [0, 0], but self.n%n_exact = {}".format(self.n%n_exact))
    
    n_rep=self.n//n_exact
    print(n_rep)

    #Poentially paralise this:
    for i in range(n_rep[0]):
        for j in range(n_rep[1]):
            # 1. Create Ising object for subsystem
            # 2. then use diagonalise()
            # 3. Add cirq.MatrixGate to self.circuit

            #maybe can write all in 1 loop by min(.., self.n[0]-self.boundaries[0] )
            #-> works already
            temp_ising = IsingDummy("GridQubit",
                                n_exact,
                                self.j_v[  i*n_exact[0]:(i+1)*n_exact[0]-b_exact[0],
                                            j*n_exact[1]: (j+1)*n_exact[1]],
                                self.j_h[ i*n_exact[0]:(i+1)*n_exact[0],
                                            j*n_exact[1]:(j+1)*n_exact[1]-b_exact[1]],
                                self.h[i*n_exact[0]:(i+1)*n_exact[0],
                                       j*n_exact[1]: (j+1)*n_exact[1]],
                                self.field)
            temp_ising.diagonalise(solver = "scipy", solver_options={"subset_by_index": [0, 2**(n_exact[0]*n_exact[1]) - 1]})
            #print("temp_ising.hamiltonian: \t {},\ntemp_ising.hamiltian.matrix() \n {} ,\ntemp_ising.eig_vec() \n {} \n"\
            #    .format(temp_ising.hamiltonian, temp_ising.hamiltonian.matrix(), temp_ising.eig_vec))

            yield cirq.MatrixGate(temp_ising.eig_vec).\
                on(*list(itertools.chain(*self.qubits[i*n_exact[0]:(i+1)*n_exact[0]][j*n_exact[1]: (j+1)*n_exact[1]])))
    if (self.n == n_exact).all() :
        self.eig_val = temp_ising.eig_val
        self.eig_vec = temp_ising.eig_vec

def _hadamard_layer(self):
    for row in self.qubits:
        for qubit in row:
            yield cirq.H.on(qubit)

def add_missing_cpv(self):
    _add_vec = []
    
    #1. Find all circuit parameter|variables that are used
    for moment in self.circuit.moments:
        #print(moment.__dict__)
        for operation in moment._operations:
            #print(operation._gate.__dict__.values())
            symbols = list(operation._gate.__dict__.values())
            for element in symbols:
                try:
                    symbol = next(iter(element.atoms(sympy.Symbol)))
                    #print(symbol)
                    if symbol not in self.circuit_param:
                        _add_vec.append(symbol)
                    #print("_add_vec: \t {}".format(_add_vec))
                except:
                    pass 
                #print(next(iter(symbol)) )
    #print("_add_vec: \t {}".format(_add_vec))
    self.circuit_param.extend(_add_vec)
    self.circuit_param_values = np.append(self.circuit_param_values , [0]*len(_add_vec))

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

    if self.basics.options["start"] == None:
        pass
    elif self.basics.options["start"] == "exact":
        self.circuit.insert(0,self.basics._exact_layer(self))
    elif self.basics.options["start"] == "hadamard":
        self.circuit.insert(0,self.basics._hadamard_layer(self))
    elif self.basics.options["start"] == "neel":
        self.circuit.insert(0,self.basics._neel_layer(self))
    elif self.basics.options["start"] == "mf":
        self.circuit.insert(0,self.basics._mf_layer(self))
    else:
        assert (False), "Invalid self.basics option, received: '{}', allowed is \n \
                'exact', 'hadamard', 'neel',  and 'mf'".format(self.basics.options["start"] )

    if self.basics.options["end"] == None:
        pass
    elif self.basics.options["end"] == "exact":
        self.circuit.append(self.basics._exact_layer(self))
    elif self.basics.options["end"] == "hadamard":
        self.circuit.append(self.basics._hadamard_layer(self))
    elif self.basics.options["end"] == "neel":
        self.circuit.append(self.basics._neel_layer(self))
    elif self.basics.options["end"] == "mf":
        self.circuit.append(self.basics._mf_layer(self))
    else:
        assert (False), "Invalid self.basics option, received: '{}', allowed is \n \
                'exact', 'hadamard', 'neel',  and 'mf'".format(self.basics.options["end"]  )



class IsingDummy(AbstractModel):
    """
    2D Ising Dummy class in order to be able to use it in 
    Ising submodules basics, hea, qaoa

    Better structure might be:

                    IsingAbstract(AbstractModel)
                        |               |
        Ising(IsingAbstract)        IsingDummy(IsingAbstract)
    """
    def __init__(self, qubittype, n, j_v, j_h, h, field: Literal["Z", "X"] = "X", t: Real = 0):
        """
        qubittype as defined in AbstractModel
        n number of qubits
        j_v vertical j's
        j_h horizontal j's
        h  strength external field
        field: basis of external field X or Z
        """
        # convert all input to np array to be sure
        super().__init__(qubittype, np.array(n))
        self.circuit_param = None
        self._set_jh(j_v, j_h, h)
        self.field = field
        self._set_hamiltonian()
        super().set_simulator()
        self.t = t

    def _set_jh(self, j_v, j_h, h):
        # convert input to numpy array to be sure
        j_v = np.array(j_v)
        # J vertical needs one row/horizontal line less
        # NEED FOR IMPROVEMENT
        assert (j_v.shape == (self.n - np.array((1, 0)))).all() or (
            j_v.shape == self.n
        ).all(), "Error in Ising._set_jh(): j_v.shape != n - {{ (1,0), (0,0)}}, {} != {}".format(
            j_v.shape, (self.n - np.array((1, 0)))
        )
        self.j_v = j_v

        # convert input to numpy array to be sure
        j_h = np.array(j_h)
        # J horizontal needs one column/vertical line less#
        # NEED FOR IMPROVEMENT
        assert (j_h.shape == (self.n - np.array((0, 1)))).all() or (
            j_h.shape == self.n
        ).all(), "Error in Ising._set_jh(): j_h.shape != n - {{ (0,1), (0,0)}}, {} != {}".format(
            j_h.shape, (self.n - np.array((0, 1)))
        )
        self.j_h = j_h

        # Set boundaries:
        self.boundaries = np.array((self.n[0] - j_v.shape[0], self.n[1] - j_h.shape[1]))

        # convert input to numpy array to be sure
        h = np.array(h)
        assert (
            h.shape == self.n
        ).all(), "Error in Ising._set_jh():: h.shape != n, {} != {}".format(h.shape, self.n)
        self.h = h

    def _set_hamiltonian(self, reset: bool = True):
        """
            Append or Reset Hamiltonian

            Create a cirq.PauliSum object fitting to j_v, j_h, h  
        """
        if reset:
            self.hamiltonian = cirq.PauliSum()

        #Conversion currently necessary as numpy type * cirq.PauliSum fails
        j_v = self.j_v.tolist()
        j_h = self.j_h.tolist()
        h = self.h.tolist()
        
        print(self.n)
        # 1. Sum over inner bounds
        for i in range(self.n[0] - 1):
            for j in range(self.n[1] - 1):
                #print("i: \t{}, j: \t{}".format(i,j))
                self.hamiltonian -= j_v[i][j]*cirq.Z(self.qubits[i][j])*cirq.Z(self.qubits[i+1][j])
                self.hamiltonian -= j_h[i][j]*cirq.Z(self.qubits[i][j])*cirq.Z(self.qubits[i][j+1])

        for i in range(self.n[0] - 1):
            j = self.n[1] - 1
            self.hamiltonian -= j_v[i][j]*cirq.Z(self.qubits[i][j])*cirq.Z(self.qubits[i+1][j])

        for j in range(self.n[1] - 1):
            i = self.n[0] - 1
            self.hamiltonian -= j_h[i][j]*cirq.Z(self.qubits[i][j])*cirq.Z(self.qubits[i][j+1])
 
        
        #2. Sum periodic boundaries
        if self.boundaries[1] == 0:
            for i in range(self.n[0]):
                j = self.n[1] - 1
                self.hamiltonian -= j_h[i][j]*cirq.Z(self.qubits[i][j])*cirq.Z(self.qubits[i][0])

        if self.boundaries[0] == 0:
            for j in range(self.n[1]):
                i = self.n[0] - 1
                self.hamiltonian -= j_v[i][j]*cirq.Z(self.qubits[i][j])*cirq.Z(self.qubits[0][j])
        
        # 3. Add external field
        if self.field == "X":
            field_gate = cirq.X
        elif self.field == "Z":
            field_gate = cirq.Z

        for i in range(self.n[0]):
            for j in range(self.n[1]):
                self.hamiltonian -= h[i][j]*field_gate(self.qubits[i][j])

    def copy(self): pass
    def energy(self): pass 
    def from_json_dict(self): pass 
    def to_json_dict(self): pass