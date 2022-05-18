"""
    This submodule creates basic circuits:
        -subsystem rotation circuit "exact"
        -Hadamrd layer "hadamard", use toi change Z <-> X basis
        -Identity layer on each qubit to include them in the circuit "identity"
        -Ising meanfield layer "mf"
        -Ising neel layer "neel"

    To Do:
        Possibly migrate subsystem rotation circuits to own submodule
"""
# external import
import cirq
from numbers import Real
import numpy as np
import sympy
from typing import Literal, List, Union
import warnings

#Internal import
from fauvqe.models.abstractmodel import AbstractModel

def set_circuit(self):
    if self.basics.options.get('append') is False:
        self.circuit = cirq.Circuit()

    tmp = self.basics.options.get("start")
    if tmp is None:
        pass
    elif tmp == "exact":
        self.circuit.insert(0,self.basics._exact_layer(self))
    elif tmp == "hadamard":
        self.circuit.insert(0,self.basics._hadamard_layer(self))
    elif tmp == "identity":
        self.circuit.insert(0,self.basics._identity_layer(self))
    elif tmp =="mf":
        self.circuit.insert(0,self.basics._mf_layer(self))
    elif tmp == "neel":
        self.circuit.insert(0,self.basics._neel_layer(self))
    else:
        assert (False), "Invalid self.basics option, received: '{}', allowed is \n \
                'exact', 'hadamard', 'identity', 'mf' and 'neel'".format(self.basics.options.get("start") )

    tmp = self.basics.options.get("end")
    if tmp is None:
        pass
    elif tmp == "exact":
        self.circuit.append(self.basics._exact_layer(self))
    elif tmp == "hadamard":
        self.circuit.append(self.basics._hadamard_layer(self))
    elif tmp == "identity":
        self.circuit.append(self.basics._identity_layer(self))
    elif tmp == "mf":
        self.circuit.append(self.basics._mf_layer(self))
    elif tmp == "neel":
        self.circuit.append(self.basics._neel_layer(self))
    else:
        assert (False), "Invalid self.basics option, received: '{}', allowed is \n \
                'exact', 'hadamard', 'identity', 'mf' and 'neel'".format(self.basics.options.get("end")  )

    #Match case the python version of switch case only works
    #starting with Python 3.10
    #match self.basics.options.get("start"):
    #    case None:
    #        pass
    #    case "exact":
    #        self.circuit.insert(0,self.basics._exact_layer(self))
    #    case "hadamard":
    #        self.circuit.insert(0,self.basics._hadamard_layer(self))
    #    case "identity":
    #        self.circuit.insert(0,self.basics._identity_layer(self))
    #    case "mf":
    #        self.circuit.insert(0,self.basics._mf_layer(self))
    #    case "neel":
    #        self.circuit.insert(0,self.basics._neel_layer(self))
    #    case _:
    #        assert (False), "Invalid self.basics option, received: '{}', allowed is \n \
    #            'exact', 'hadamard', 'identity', 'mf' and 'neel'".format(self.basics.options.get("start") )

    #match self.basics.options.get("end"):
    #    case None:
    #        pass
    #    case "exact":
    #        self.circuit.append(self.basics._exact_layer(self))
    #    case "hadamard":
    #        self.circuit.append(self.basics._hadamard_layer(self))
    #    case "identity":
    #        self.circuit.append(self.basics._identity_layer(self))
    #    case "neel":
    #        self.circuit.append(self.basics._neel_layer(self))
    #    case "mf":
    #        self.circuit.append(self.basics._mf_layer(self))
    #    case _:
    #        assert (False), "Invalid self.basics option, received: '{}', allowed is \n \
    #            'exact', 'hadamard', 'identity', 'mf' and 'neel'".format(self.basics.options.get("end")  )

def _exact_layer(self):
    """
    This function creates a circuit to rotate from the computational Z basis
    into a subsystem eigenbasis.

    The default behaviour is:
        Use a fitting subsystem size and cut the bounds between the subsystems.
        In Particular:
        1. Check whether given subgrid is multiple of overall grid
        2. Use this to determine where/ how to apply it
        3. Need to generate and solve smaller proxy systems
    
        Example of default behaviour: Cut 2x4 into 1x2 subsystems
            x - x - x - x -     x - x, x - x -,
            |   |   |   |    => x - x or x - x - 
            x - x - x - x -     |   |    |   |  
        
        hard to tell what's best => further flag
        In the default behaviour the boundary conditions are [1,1] meaning closed boundaries

    Parameters
    ----------
        b_exact: [int,int]
            Boundary conditions of the subsystems

        cc_exact: boolean
            If true return the hermitian conjugate 
            Usually needed if a given state should be expressed in subsystem basis

        n_exact: [int,int]
            System size of the subsystems

        SingleQubitGates: List[cirq SingleQubiteGates]
            The default uses the SingleQubitGates from SpinModel. 
            However, it can be set here seperately to rotate in a subsystem basis that e.g. has only parts of the systems singel Qubit terms

        subsystem_h: List[3D numpy arrays]:
            Requires subsystem_qubits
            Allows to split SingleQubitTerms in multiple subsystem coverings to realise matching term counting

        subsystem_j_h: List[3D numpy arrays]:
            Requires subsystem_qubits
            Allows to split TwoQubitTerms in multiple subsystem coverings to realise matching term counting

        subsystem_j_v: List[3D numpy arrays]:
            Requires subsystem_qubits
            Allows to split TwoQubitTerms in multiple subsystem coverings to realise matching term counting

        subsystem_qubits: List[List[cirq Qubits]]
            Generalisation of n_exact, this allows to have non symmetric coveraging. e.g. cuting 2x4 into 2x1, 2x2 and 1x2

        TwoQubitGates: List[cirq TwoQubiteGates]
            The default uses the TwoQubitGates from SpinModel. 
            However, it can be set here seperately to rotate in a subsystem basis that e.g. has only parts of the systems two Qubit terms


    To Dos
    ----------
        -Implement functionality to create subsystem rotations from "subsystem_hamiltonians" (cirq.PauliSums)
        -Make subsystem_h, subsystem_j_h and subsystem_j_v work with n_exact (currently require subsystem_qubits)
        -When subsystem_qubits are used, allow for b_exact != [1,1] (so far assumed/used as simplification)
        -More uniform naming of flags
        -Simplify inputs for SpinModels with only 1 Single or TwoQubitGate -> needs to be done first in SpinModel

    """
    if self.qubittype != "GridQubit":
        raise NotImplementedError()

    #Init self.subsystem_energies: List[nd.arrays] if it does not exist
    #Store subsystem energy spectrum as need for energy filter
    if hasattr(self, 'subsystem_energies') is False or self.basics.options.get("append") is False:
        self.subsystem_energies = []

    if hasattr(self, 'subsystem_U') is False or self.basics.options.get("append") is False:
        self.subsystem_U = []

    #Init self.subsystem_hamiltonians: List[cirq.PauliSum] if it does not exist
    if hasattr(self, 'subsystem_hamiltonians') is False or self.basics.options.get("append") is False:
        self.subsystem_hamiltonians = []

    #Init self.subsystem_qubits: List[List[cirq.GridQubit]]
    if hasattr(self, 'subsystem_qubits') is False or self.basics.options.get("append") is False:
        self.subsystem_qubits = []

    #Get b_exact or set default
    if self.basics.options.get("b_exact") is None:
        b_exact = [1,1]
    else:
        b_exact = self.basics.options.get("b_exact")

    #Get SingleQubitGates or set default
    if self.basics.options.get("SingleQubitGates") is None:
        SingleQubitGates = self._SingleQubitGates,
    else:
        SingleQubitGates = self.basics.options.get("SingleQubitGates")

    #Get TwoQubitGatesor set default
    if self.basics.options.get("TwoQubitGates") is None:
        TwoQubitGates =self._TwoQubitGates,
    else:
        TwoQubitGates = self.basics.options.get("TwoQubitGates")

    #If subsystem_qubits is None fall back to using n_exact
    if self.basics.options.get("subsystem_qubits") is None and self.basics.options.get("subsystem_hamiltonians") is None:
        n_exact = self.basics.options.get("n_exact")

        if np.sum(self.n%n_exact) != 0:
            warnings.warn("IsingBasicsWarning: self.n%n_exact != [0, 0], but self.n%n_exact = {}".format(self.n%n_exact))
        
        n_rep=self.n//n_exact
        #print("n_rep: \t {}".format(n_rep))

        #Poentially paralise this:
        for i in range(n_rep[0]):
            for j in range(n_rep[1]):
                # 1. Create Ising object for subsystem
                # 2. then use diagonalise()
                # 3. Add cirq.MatrixGate to self.circuit

                #np.transpose here required due to bad implementation in SpinModel
                #Very unexpected behaviour + slows down coding
                temp_model = SpinModelDummy("GridQubit",
                                    n_exact,
                                    np.transpose(self.j_v[  i*n_exact[0]:(i+1)*n_exact[0]-b_exact[0],
                                                            j*n_exact[1]: (j+1)*n_exact[1],:], 
                                                            (2, 0,1)),
                                    np.transpose(self.j_h[  i*n_exact[0]:(i+1)*n_exact[0],
                                                            j*n_exact[1]:(j+1)*n_exact[1]-b_exact[1], :], 
                                                            (2, 0,1)),
                                    np.transpose(self.h[i*n_exact[0]:(i+1)*n_exact[0],
                                                        j*n_exact[1]: (j+1)*n_exact[1],:], 
                                                        (2, 0,1)),
                                    *TwoQubitGates,
                                    *SingleQubitGates,
                                    n_offset=[i*n_exact[0],j*n_exact[1]])

                #Store cirq PauliSums of subsystem Hamiltonians
                self.subsystem_hamiltonians.append(temp_model.hamiltonian)

                if self.basics.options.get("subsystem_diagonalisation") is not False:
                    #To not loose qubits with no gates/ 0 qubits
                    temp_matrix = temp_model.hamiltonian.matrix(flatten(temp_model.qubits))

                    temp_model.diagonalise( solver = "scipy", 
                                            solver_options={"subset_by_index": [0, 2**(n_exact[0]*n_exact[1]) - 1]},
                                            matrix= temp_matrix)
                    
                    self.subsystem_energies.append(temp_model.eig_val)
                    #This would be nicer in 1 line, but 2D list slicing in python 
                    #Resulted in wrong result. compare lab book 2 page 109
                    temp_qubits = []
                    for k in range(n_exact[0]):
                        for l in range(n_exact[1]):
                            temp_qubits.append(self.qubits[i*n_exact[0]+k][j*n_exact[1]+l])
                    #print("temp_qubits: \t {}".format(temp_qubits))

                    #Store subsystem qubits
                    self.subsystem_qubits.append(temp_qubits)

                    #Get cc_exact or set default
                    if self.basics.options.get("cc_exact") is True:
                        yield cirq.MatrixGate(  np.matrix.getH(temp_model.eig_vec),
                                                unitary_check_rtol=1e-12,
                                                unitary_check_atol=1e-12,
                                                ).on(*temp_qubits)
                    else:
                        yield cirq.MatrixGate(  temp_model.eig_vec,
                                                unitary_check_rtol=1e-12,
                                                unitary_check_atol=1e-12,
                                                ).on(*temp_qubits)
    else:
        #More flexible option
        #give lists of GridQubits to devide the system
        #allow to adapt J and h of subsystem to prevent overcounting by multiple coverage
        # if subsystem_j_v, subsystem_j_h or subsystem_h are not given use self.j_v, self.j_h or self.h accordingly
         #ToDo: apply b_exact accordingly
        
        #Check whether set of qubits match and non is there twice
        subsystem_qubits = self.basics.options.get("subsystem_qubits")
        assert ( set(self.basics.flatten(self.qubits)) == set(self.basics.flatten(subsystem_qubits)) 
            ), "Subsystem qubits do not match system qubits;\nProvided system qubits are:\n{}\nProvided sub system qubits are:\n{}\n".format(
            set(self.basics.flatten(self.qubits)), 
            set(self.basics.flatten(subsystem_qubits))
            )
        
        #Store subsystem qubits
        self.subsystem_qubits = subsystem_qubits

        #Get subsystem_h or set default
        if self.basics.options.get("subsystem_h") is None:
            subsystem_h = []
            for i in range(len(subsystem_qubits)):
                subsystem_h.append(self.h[min(subsystem_qubits[i])._row: max(subsystem_qubits[i])._row+1,
                        min(subsystem_qubits[i])._col: max(subsystem_qubits[i])._col+1, :])
        else:
            subsystem_h = self.basics.options.get("subsystem_h")

        #Get subsystem_j_h or set default
        if self.basics.options.get("subsystem_j_h") is None:
            subsystem_j_h = []
            for i in range(len(subsystem_qubits)):
                _real_col = min(1, max(subsystem_qubits[i])._col- min(subsystem_qubits[i])._col)
                subsystem_j_h.append(self.j_h[min(subsystem_qubits[i])._row: max(subsystem_qubits[i])._row+1,
                        min(subsystem_qubits[i])._col: max(subsystem_qubits[i])._col+_real_col-b_exact[1], :])
        else:
            subsystem_j_h = self.basics.options.get("subsystem_j_h")

        #Get subsystem_j_v or set default
        if self.basics.options.get("subsystem_j_v") is None:
            subsystem_j_v = []
            for i in range(len(subsystem_qubits)):
                _real_row = min(1, max(subsystem_qubits[i])._row- min(subsystem_qubits[i])._row)
                subsystem_j_v.append(   self.j_v[   min(subsystem_qubits[i])._row: (max(subsystem_qubits[i])._row+_real_row-b_exact[0]),
                                                    min(subsystem_qubits[i])._col: max(subsystem_qubits[i])._col+1, :])
        else:
            subsystem_j_v = self.basics.options.get("subsystem_j_v")

        #Prints for debugging and to confirm correct structure of subsystem_h etc
        #Keep those for the moment
        if self.basics.options.get("print") is True:
            print("self.h:\n{}\nsubsystem_h: \n{}".format(self.h, subsystem_h))          # pragma: no cover 
            print("self.j_h:\n{}\nsubsystem_j_h: \n{}".format(self.j_h, subsystem_j_h))  # pragma: no cover 
            print("self.j_v:\n{}\nsubsystem_j_v: \n{}".format(self.j_v, subsystem_j_v))  # pragma: no cover 
        
        for i in range(len(subsystem_qubits)):
            #Need to calculate n_exact from subsystem_qubits
            n_exact = [ max(subsystem_qubits[i])._row- min(subsystem_qubits[i])._row + 1,
                        max(subsystem_qubits[i])._col- min(subsystem_qubits[i])._col + 1]
            
            #np.transpose here required due to bad implementation in SpinModel
            #Very unexpected behaviour + slows down coding
            temp_model = SpinModelDummy("GridQubit",
                                    n_exact,
                                    np.transpose(subsystem_j_v[i], (2, 0,1)),
                                    np.transpose(subsystem_j_h[i], (2, 0,1)),
                                    np.transpose(subsystem_h[i], (2, 0,1)),
                                    *TwoQubitGates,
                                    *SingleQubitGates,
                                    n_offset=[min(subsystem_qubits[i])._row, min(subsystem_qubits[i])._col] )
                                    
            #Store cirq PauliSums of subsystem Hamiltonians
            print(temp_model.hamiltonian)
            self.subsystem_hamiltonians.append(temp_model.hamiltonian)
            
            #To not loose qubits with no gates/ 0 qubits
            #TO DO: this can be much mor efficent by using converter class and scipy sparse
            if self.basics.options.get("subsystem_diagonalisation") is not False:
                temp_matrix = temp_model.hamiltonian.matrix(flatten(temp_model.qubits))
                temp_model.diagonalise( solver = "scipy", 
                                        solver_options={"subset_by_index": [0, 2**(n_exact[0]*n_exact[1]) - 1]},
                                        matrix= temp_matrix)
                #tmp_eig_val=temp_model.eig_val
                #temp_model.diagonalise( solver = "numpy", 
                #                        matrix= temp_matrix)
                #print(tmp_eig_val-temp_model.eig_val)
                #print("type(temp_model.eig_val[0]): {}".format(type(temp_model.eig_val[0])))

                self.subsystem_energies.append(temp_model.eig_val)
                self.subsystem_U.append(temp_model.eig_vec)
                #Get cc_exact or set default
                if self.basics.options.get("cc_exact") is True:
                    yield cirq.MatrixGate(  matrix=np.matrix.getH(temp_model.eig_vec),
                                            unitary_check_rtol=1e-12,
                                            unitary_check_atol=1e-12,
                                            ).on(*subsystem_qubits[i])
                else:
                    yield cirq.MatrixGate(temp_model.eig_vec,
                                        unitary_check_rtol=1e-12,
                                        unitary_check_atol=1e-12,
                                        ).on(*subsystem_qubits[i])

    #If this method is used to rotate into the eigenbasis, store eigenvalues and vectors, as those are already calcualted
    if (self.n == n_exact).all() and self.basics.options.get("subsystem_diagonalisation") is not False:
        self.eig_val = temp_model.eig_val
        self.eig_vec = temp_model.eig_vec

def _hadamard_layer(self):
    for row in self.qubits:
        for qubit in row:
            yield cirq.H.on(qubit)

def _identity_layer(self):
    for row in self.qubits:
        for qubit in row:
            yield cirq.I.on(qubit)

def _mf_layer(self):
    assert self.field == "X","Mean field layer only implemented for self.field == 'X'"
    for row in self.qubits:
        for qubit in row:
            theta = self.basics.__get_mf_angle(self, qubit)
            #print("theta: \t{}".format(theta))
            yield cirq.XPowGate(exponent=(theta/np.pi)).on(qubit)

def _neel_layer(self):
    for row in self.qubits:
        for qubit in row:
            # Possibly only specific to Grid Qubit:
            if (qubit._row + qubit._col)%2 == 0:
                yield cirq.X(qubit)

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
            for g in range(len(self._TwoQubitGates)):
                J_mean += self.j_v[qubit._row - 1,  qubit._col][g]
                n_J += 1
        
        # (4)
        if qubit._col <  self.j_v.shape[1] and qubit._row <  self.j_v.shape[0]:
            for g in range(len(self._TwoQubitGates)):
                J_mean += self.j_v[qubit._row, qubit._col][g]
                n_J += 1
    
    if (self.j_h.shape[0] * self.j_h.shape[1]) > 0: 
        # (2)
        if qubit._col > 0:
            for g in range(len(self._TwoQubitGates)):
                J_mean += self.j_h[qubit._row, qubit._col -1][g]
                n_J += 1

        # (3)
        if qubit._col <  self.j_h.shape[1] and qubit._row <  self.j_h.shape[0]:
            for g in range(len(self._TwoQubitGates)):
                J_mean += self.j_h[qubit._row, qubit._col][g]
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
                    if symbol not in self.circuit_param and\
                       symbol not in _add_vec:
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

def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

def get_energy_filter_from_subsystem(self, subsystem_energies = None, do_reorder = True):
    #To Do Add qubit_map from subsystem qubits
    #Set a subsystem qubit_map -> need later for simulations
    if hasattr(self, 'subsystem_qubits') is True and hasattr(self, 'subsystem_qubit_map') is True:
        self.subsystem_qubit_map = {}
        for i_subsystem in range(len(self.subsystem_qubits)):
            self.subsystem_qubit_map.update({self.subsystem_qubits[i_subsystem][l]: int(l + len(self.subsystem_qubit_map)) for l in range(len(self.subsystem_qubits[i_subsystem]))})

    #Do this two step if to away issue that all() not applicable for None
    #But also all() needed if subsystem_energies not None
    if np.size(subsystem_energies) == 1:
        if (subsystem_energies == None):
            if hasattr(self, 'subsystem_energies') is True:
                subsystem_energies = self.subsystem_energies
            else:
                assert False, "No subsystem energies provided"

    if np.size(subsystem_energies) == 2**np.size(self.qubits):
        return np.squeeze(subsystem_energies)
    else:
        print(subsystem_energies)
        #energy_filter = np.add(np.size( self.subsystem_qubits[0])*self.subsystem_energies[0]
        #                            .reshape((1,2**np.size( self.subsystem_qubits[0]))), 
        #                       np.size( self.subsystem_qubits[1])*self.subsystem_energies[1]
        #                            .reshape((2**np.size( self.subsystem_qubits[1]),1))
        #                        ).reshape((1,2**np.size( self.subsystem_qubits[0] + self.subsystem_qubits[1]))) 
        energy_filter = np.add(np.size( self.subsystem_qubits[1])*self.subsystem_energies[1]
                                    .reshape((1,2**np.size( self.subsystem_qubits[1]))), 
                               np.size( self.subsystem_qubits[0])*self.subsystem_energies[0]
                                    .reshape((2**np.size( self.subsystem_qubits[0]),1))
                                ).reshape((1,2**np.size( self.subsystem_qubits[0] + self.subsystem_qubits[1]))) 
        for i in range(2,len(subsystem_energies)):
            energy_filter = np.add( np.size( self.subsystem_qubits[i])*self.subsystem_energies[i]
                                        .reshape((1,2**np.size( self.subsystem_qubits[i]))),
                                    energy_filter.reshape((np.size(energy_filter),1))
                                    ).reshape((1,np.size(energy_filter)*2**np.size(self.subsystem_qubits[i]))) 
        
        # If the subsystems are not in standard order return energy filter in standard order
        ordering = self.basics.get_reordering_from_subsystem(self)
        print("ordering: {}\nnp.size(energy_filter): {}\nnp.squeeze(energy_filter)/np.size(self.qubits) {}\nself.basics.permute_state_vector: {}"
        .format(ordering, 
                np.size(energy_filter),
                np.size(np.squeeze(energy_filter)/np.size(self.qubits)),
                np.size(self.basics.permute_state_vector(   self,
                                                        np.squeeze(energy_filter)/np.size(self.qubits), 
                                                        ordering))))
        if (ordering == range(np.size(self.qubits))) or (not do_reorder):
            return np.squeeze(energy_filter)/np.size(self.qubits)
        else:
            return self.basics.permute_state_vector(   self,
                                                        np.squeeze(energy_filter)/np.size(self.qubits), 
                                                        ordering)

def get_subsystem_qubit_map(self, 
                            return_list=False):
    qubit_map={}
    for i in np.arange(len(self.subsystem_qubits)):
        #print(self.subsystem_qubits[i])
        tmp_qubit_map = {self.subsystem_qubits[i][k]: int(len(qubit_map)+ k) for k in np.arange(len(self.subsystem_qubits[i]))}
        qubit_map.update(tmp_qubit_map)
    
    #If flag is set cast dictionary to list
    if return_list:
        return list(dict(sorted(qubit_map.items(), key=lambda item: item[1])).keys())
    else:
        return qubit_map

def get_subsystem_qubit_order(self):
    #Aliase to get qubit list
    return self.basics.get_subsystem_qubit_map(self,return_list=True)

def permute_state_vector(   self, 
                            wavefunction: np.ndarray,
                            permutations: Union[List[int],List[List[int]]]):
    #Might be important to make this efficently
    # Note only differing qubits need to be swapped
    # e.g. 2 qubits:
    #   |00>    ->  |00>    NO Swap  
    #   |01>    ->  |10>    Swap
    #   |10>    ->  |01>    Swap
    #   |11>    ->  |11>    NO Swap 
    # probably easiest ist to just swap all 
    # 1. get old/new indices list
    # 2. wavefunction[new_ind] = wavefunction[old_ind]
    # or wavefunction = wavefunction[new_order_Hilbert]
    _n = int(np.log2(np.size(wavefunction)))
    new_indices=np.arange(2**_n)

    if all(isinstance(i, int) for i in permutations):
        permutations = [permutations]

    for permutation in permutations:
        non_permuted = sorted(permutation)
        
        bin_indices = ((new_indices.reshape(-1,1) & (2**np.arange(_n))) != 0).astype(np.int64)
        bin_indices[:,non_permuted]= bin_indices[:,permutation]

        if _n <= 8:
            #This works for n<9:
            #reason backbits works only fo uInt8
            new_indices = np.squeeze(np.packbits(bin_indices,axis=1,bitorder='little'))
        else:
            new_indices = np.dot(   np.squeeze(np.packbits(bin_indices,axis=1,bitorder='little').view(np.uint8)), 
                                2**np.array([8*i for i in range(int(np.ceil(_n/8)))]))
        #print("np.size(bin_indices): {}\nnp.size(new_indices): {}".format(np.size(bin_indices),np.size(new_indices)))

    return wavefunction[new_indices]

def get_reordering_from_subsystem(self):
    # from assert in exact_layer know that qubits and subsystems are the same
    #Get qubit flatten list
    _system_qubits= self.basics.flatten(self.qubits)

    #Get subsystem qubits flatten list
    _subsystem_qubits= self.basics.flatten(self.subsystem_qubits)

    #Return index reordering
    return [_system_qubits.index(_subsystem_qubits[i]) for i in range(len(_system_qubits))]

class SpinModelDummy(AbstractModel):
    """
    2D Ising Dummy class in order to be able to use it in 
    Ising submodules basics, hea, qaoa

    Better structure might be:

                    SpinModelAbstract(AbstractModel)
                        |               |
        SpinModel(IsingAbstract)        SpinModelDummy(IsingAbstract)
    """
    def __init__(self, 
                 qubittype, 
                 n,
                 j_v,
                 j_h,
                 h,
                 TwoQubitGates: List[cirq.PauliSum],
                 SingleQubitGates: List[cirq.PauliSum],
                 t: Real = 0,
                 n_offset = [0,0]):
        """
        qubittype as defined in AbstractModel
        n number of qubits
        j_v vertical j's - same order as TwoQubitGates
        j_h horizontal j's - same order as TwoQubitGates
        h  strength external fields - same order as SingleQubitGates
        TwoQubitGates: list of 2 Qubit Gates
        SingleQubitGates: list of single Qubit Gates
        t: Simulation Time
        """
        # convert all input to np array to be sure
        super().__init__(qubittype, np.array(n))
        
        self._offset_qubits(n_offset)
        
        self.circuit_param = None
        self.circuit_param_values = np.array([])
        self._TwoQubitGates = TwoQubitGates
        self._SingleQubitGates = SingleQubitGates
        self._set_jh(j_v, j_h, h, TwoQubitGates, SingleQubitGates)
        self._set_hamiltonian()
        super().set_simulator()
        self.t = t

    def _offset_qubits(self, n_offset):
        temp = [[cirq.GridQubit(i, j) for j in range(n_offset[1],n_offset[1]+self.n[1])] for i in range(n_offset[0],n_offset[0]+self.n[0])]
        self.qubits = temp

    def _set_jh(self, j_v, j_h, h, TwoQubitGates, SingleQubitGates):
        # convert input to numpy array to be sure
        j_v = np.array(j_v)
        # J vertical needs one row/horizontal line less
        # NEED FOR IMPROVEMENT
        assert (j_v.shape == (len(TwoQubitGates), *(self.n - np.array((1, 0))) )) or (
            j_v.shape == (len(TwoQubitGates), *self.n)
        ), "Error in SpinModel._set_jh(): j_v.shape != (len(TwoQubitGates), n - {{ (1,0), (0,0)}}), {} != {}".format(
            j_v.shape, (len(TwoQubitGates), *(self.n - np.array((1, 0))))
        )
        self.j_v = np.transpose(j_v, (1, 2, 0))
        
        # convert input to numpy array to be sure
        j_h = np.array(j_h)
        # J horizontal needs one column/vertical line less#
        # NEED FOR IMPROVEMENT
        assert (j_h.shape == (len(TwoQubitGates), *(self.n - np.array((0, 1))) )) or (
            j_h.shape == (len(TwoQubitGates), *self.n)
        ), "Error in SpinModel._set_jh(): j_h.shape != (len(TwoQubitGates), n - {{ (1,0), (0,0)}}), {} != {}".format(
            j_h.shape, (len(TwoQubitGates), *(self.n - np.array((1, 0))))
        )
        self.j_h = np.transpose(j_h, (1, 2, 0))

        # Set boundaries:
        self.boundaries = np.array((self.n[0] - j_v.shape[1], self.n[1] - j_h.shape[2]))

        # convert input to numpy array to be sure
        h = np.array(h)
        assert (
            h.shape == (len(SingleQubitGates), *self.n)
        ), "Error in SpinModel._set_jh():: h.shape != (len(SingleQubitGates), n), {} != {}".format(h.shape, (len(SingleQubitGates), *self.n))
        self.h = np.transpose(h, (1, 2, 0))

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
        
        # 1. Sum over inner bounds
        for g in range(len(self._TwoQubitGates)):
            for i in range(self.n[0] - 1):
                for j in range(self.n[1] - 1):
                    #print("i: \t{}, j: \t{}".format(i,j))
                    self.hamiltonian -= j_v[i][j][g]*self._TwoQubitGates[g](self.qubits[i][j], self.qubits[i+1][j])
                    self.hamiltonian -= j_h[i][j][g]*self._TwoQubitGates[g](self.qubits[i][j], self.qubits[i][j+1])
        
        for g in range(len(self._TwoQubitGates)):
            for i in range(self.n[0] - 1):
                j = self.n[1] - 1
                self.hamiltonian -= j_v[i][j][g]*self._TwoQubitGates[g](self.qubits[i][j], self.qubits[i+1][j])
        
        for g in range(len(self._TwoQubitGates)):
            for j in range(self.n[1] - 1):
                i = self.n[0] - 1
                self.hamiltonian -= j_h[i][j][g]*self._TwoQubitGates[g](self.qubits[i][j], self.qubits[i][j+1])
        
        #2. Sum periodic boundaries
        if self.boundaries[1] == 0:
            for g in range(len(self._TwoQubitGates)):
                for i in range(self.n[0]):
                    j = self.n[1] - 1
                    self.hamiltonian -= j_h[i][j][g]*self._TwoQubitGates[g](self.qubits[i][j], self.qubits[i][0])
        
        if self.boundaries[0] == 0:
            for g in range(len(self._TwoQubitGates)):
                for j in range(self.n[1]):
                    i = self.n[0] - 1
                    self.hamiltonian -= j_v[i][j][g]*self._TwoQubitGates[g](self.qubits[i][j], self.qubits[0][j])
        
        # 3. Add external field
        for g in range(len(self._SingleQubitGates)):
            for i in range(self.n[0]):
                for j in range(self.n[1]):
                    self.hamiltonian -= h[i][j][g]*self._SingleQubitGates[g](self.qubits[i][j])

    def copy(self): pass
    def energy(self): pass 
    def from_json_dict(self): pass 
    def to_json_dict(self): pass