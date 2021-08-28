"""
#  TP2 internal VQE class
#   purpose is to write common code in a compatible manner
#
#  try to use type definitions and numpy arrays as good as possible
#
# use this:
# https://quantumai.google/cirq/tutorials/educators/qaoa_ising
# as a starting point.
# Write in class strucutre. Add automated testing. Put to package.
# Then add own ideas and alternative optimisers, ising circuits etc.

"""
from __future__ import annotations

import abc
from typing import Tuple, List,Optional
from numbers import Number, Real

import numpy as np
import sympy
import cirq
import qsimcirq
import timeit
#import fastmat

from scipy.linalg import eigh as scipy_solver
from scipy.sparse.linalg import eigsh as scipy_sparse_solver
from scipy.sparse import dia_matrix as scipy_dia_matrix 

from fauvqe.restorable import Restorable

class AbstractModel(Restorable):
    """
    The idea is to write a common VQE framework to which all
    our code fits so we can easily use bits and pieces from one
    another.
    """

    # Define global proberties here
    # Watch out, these are shared by all class objects, what one usually
    # wants to avoid

    def __init__(self, qubittype, n):
        """
        Write in a flag style to be insensitive for input order
        We want to set
            qubittype e.g. 'GridQubit'
            n = number of qubits, potentially array
            qubits = array of qubits
            initialise empty cirq circuit
            ######
            later intial state?
        """
        self.circuit = cirq.Circuit()
        self.circuit_param: List[sympy.Symbol] = []
        self.circuit_param_values: Optional[np.ndarray] = None
        self.hamiltonian = cirq.PauliSum()
        self.init_qubits(qubittype, n)
        self.set_simulator()
        self.t : Real = 0

        self.eig_val: Optional[np.ndarray] = None
        self.eig_vec: Optional[np.ndarray] = None
        self._Ut: Optional[np.ndarray] = None

    # initialise qubits or device
    def init_qubits(self, qubittype, n):

        # cannot use switcher as initialisation parameters 'n' are of different type
        if qubittype == "NamedQubit":
            assert all(
                isinstance(x, str) for x in n
            ), "Error in qubit initialisation: n needs to be string list for NameQubit, received: n = {}, {}".format(
                n, type(n)
            )
            # need this awkward return scheme to get right format
            # try:
            temp = [cirq.NamedQubit(x) for x in n]
            self.qubittype = "NamedQubit"
            self.n = n
            self.qubits = temp
            # With assert not needed?
            # except:
            #    print("NameQubit needs string list as input, received: {}, {}".format(n, type(n)))
        elif qubittype == "LineQubit":
            assert (
                isinstance(n, (int, np.int_)) and n > 0
            ), "Error in qubit initialisation: n needs to be natural Number for LineQubit, received: n = {}, {}".format(
                n, type(n)
            )
            # need this awkward return scheme to get right format
            # try:
            temp = [q for q in cirq.LineQubit.range(n)]
            self.qubittype = "LineQubit"
            self.n = n
            self.qubits = temp
            # With assert not needed?
            # except:
            #     print("LineQubit needs natural number as input, received: {}, {}".format(n, type(n)))
        elif qubittype == "GridQubit":
            # Potential Issue for NISQ algorithms:
            # This allows not only NN-gates, but e.g. also between
            # (0,0) and (1,1) or (2,0)
            # one might want to avoid this....
            # Solution:
            #   see page 36 cirq 0.9.0dev manual, make custom device
            # Issue: for n = np.array([1, 1]) isinstance(n[1], int) is false as np.int64
            assert (
                np.size(n) == 2
                and isinstance(n[0], (int, np.int_))
                and n[0] > 0
                and isinstance(n[1], (int, np.int_))
                and n[1] > 0
            ), "Error in qubit initialisation: n needs to be 2d-int for GridQubit, received: n = {}, {}".format(
                n, type(n)
            )
            # need this awkward return scheme to get right format
            # try:
            temp = [[cirq.GridQubit(i, j) for j in range(n[1])] for i in range(n[0])]
            self.qubittype = "GridQubit"
            self.n = n
            self.qubits = temp
            # With assert not needed?
            # except:
            #    print("GridQubit needs natural number as input, received: {}, {}".format(n, type(n)))
        else:
            assert (
                False
            ), "Invalid qubittype, received: '{}', allowed is \n \
                'NamedQubit', 'LineQubit', 'GridQubit'".format(
                qubittype
            )
        """    
            Later add also google decives here
            e.g.
                    switcher = {
        #This neglects a bit what other properties devices have
            'Bristlecone':cirq.google.Bristlecone.qubits,
            'Sycamore':   cirq.google.Sycamore.qubits,
            'Sycamore23': cirq.google.Sycamore23.qubits,
            'Foxtail':    cirq.google.Foxtail.qubits,
        }
            self.qubits = switcher.get(qubittype, "Invalid qubittype");
        
            Then get something like:
            self.qubits = 
            frozenset({cirq.GridQubit(0, 5), cirq.GridQubit(0, 6),..})
            Issue: cannot handle this quite as GridQubits, LineQubits or NameQubits
        """

    # set simualtor to be written better, aka more general
    def set_simulator(self, simulator_name="qsim", simulator_options: dict = {}, dtype = np.complex64):
        if simulator_name == "qsim":
            """
            Possible qsim options:
                Used/Usful options:
                't' : number of threads; default 't' 1
                'f': fused gate, e.g. 'f': 4 fused gates to 4-qubit gates
                        this can save MemoryBandwidth for more required calculations;
                        default 'f': 2
                qsimh options (Feynman simulator):
                Simulate between pre and suffix gates and sum over all
                pre and suffix gates
                'k': gates on the cut;default 0
                'w': ?;default 0
                'v': ? ;default 0
                'p': number of prefix gates;default 0
                'r': number of root gates;default 0
                'b':    bitstring
                'i':    ?
                'c':    ?
                'ev'. parallel used for sample expectation values?
                #'s': suffix gates p+r+s=k

            More details: https://github.com/quantumlib/qsim

            From https://github.com/quantumlib/qsim/blob/master/qsimcirq/qsimh_simulator.py:
            def __init__(self, qsimh_options: dict = {}):
                self.qsimh_options = {'t': 1, 'f': 2, 'v': 0}
                self.qsimh_options.update(qsimh_options)
            
            cirq Simulator is configured via optional arguments sim_args:
                dtype: Type[np.number] = np.complex64,
                noise: "cirq.NOISE_MODEL_LIKE" = None,
                seed: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None,
                split_untangled_states: bool = True
            """
            self.simulator_options = {"t": 8, "f": 4}
            self.simulator_options.update(simulator_options)
            self.simulator = qsimcirq.QSimSimulator(self.simulator_options)
        elif simulator_name == "cirq":
            self.simulator_options = {}
            self.simulator = cirq.Simulator(dtype=dtype)
        else:
            assert False, "Invalid simulator option, received {}, allowed is 'qsim', 'cirq'".format(
                simulator_name
            )

    def get_param_resolver(self, temp_cpv):
        joined_dict = {
            **{str(self.circuit_param[i]): temp_cpv[i] for i in range(len(self.circuit_param))}
        }

        return cirq.ParamResolver(joined_dict)

    @abc.abstractmethod
    def copy(self) -> AbstractModel:
        raise NotImplementedError()  # pragma: no cover

    @abc.abstractmethod
    def energy(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()  # pragma: no cover

    @abc.abstractmethod
    def _set_hamiltonian(self, reset: bool = True):
        raise NotImplementedError()  # pragma: no cover

    def diagonalise(self, solver = "scipy.sparse", solver_options: dict = {}):
        """
            Implementation of an exact solver for an AbstractModel object.

            This works locally for up to 14 qubits and 
            on the AMD 7502 EPYC nodes for up to 16 qubits (approx 45 min.)

            Eigen-Value-Problem (EVP) solver

            This method implements as objective the expectation value of the energies
            of the linked model.

            Parameters
            ----------
            self.eig_val:    eigen values, normalised by qubit number to keep compareability
            self.eig_vec:    eigen vector

            different solvers, possibly use import lib
                "numpy"         np.linalg.eigh
                "scipy"         scipy.linalg.eigh
                "scipy.sparse"  scipy.sparse.linalg.eigsh
                + Add more hardwareefficent solver
                + Set usful defaults. 
            parameters to pass on to the solver
                implement as dict
                e.g. k = 2
        """
        __n = np.size(self.qubits)
        if solver == "numpy":
            self.eig_val, self.eig_vec =  np.linalg.eigh(self.hamiltonian.matrix())
            # Normalise eigenvalues
            self.eig_val /= __n   
            # Normalise eigenvectors ?      
        elif solver == "scipy":
            self.solver_options = { "check_finite": False, 
                                    "subset_by_index": [0, 1]}
            self.solver_options.update(solver_options)           
            self.eig_val, self.eig_vec = scipy_solver(
                self.hamiltonian.matrix(), 
                **self.solver_options,
                )
            # Normalise eigenvalues
            self.eig_val /= __n
            # Normalise eigenvectors ?
        elif solver == "scipy.sparse":
            self.solver_options = { "k": 2,
                                    "which": 'SA'}
            self.solver_options.update(solver_options)
            self.eig_val, self.eig_vec = scipy_sparse_solver(
                self.hamiltonian.matrix(), 
                **self.solver_options,
                )
            # Normalise eigenvalues
            self.eig_val /= __n
            # Normalise eigenvectors ?
        else:
            assert False, "Invalid simulator option, received {}, allowed is 'numpy', 'scipy', 'scipy.sparse'".format(
                solver
            )

    def set_Ut(self, use_dense: bool = False):
        #https://docs.sympy.org/latest/modules/numeric-computation.html
        #1.If exact diagonalisation exists already, don't calculate it again
        # Potentially rather use scipy here!
        _n = np.size(self.qubits)
        _N = 2**(_n)
        
        if self.t == 0:
            self._Ut = np.identity(_N)
            return True
        
        if np.size(self.eig_val) != _N or \
        (np.shape(self.eig_vec) != np.array((_N, _N)) ).all():
            self.diagonalise(solver = "scipy", solver_options={"subset_by_index": [0, _N - 1]})
        
        #print("eig_val: \t {}, eig_vec \t {}, _N \t {}".\
        #                format(np.size(self.eig_val), np.shape(self.eig_vec) ,_N))
        
        #2. The do exact diagonlisation + use sympy symbol
        # U(t) = (v1 .. vN) diag(e⁻iE1t .. e-iENt) (v1 .. vN)^+
        
        #t0 = timeit.default_timer()
        # by *_n account for eigen values are /_n but want time evolution of actual Hamiltonian
        if (np.size(self.qubits) < 12) or use_dense :
            self._Ut = np.matmul(np.matmul(self.eig_vec,np.diag(np.exp(-1j*_n*self.eig_val*self.t)), dtype = np.complex128),
                            self.eig_vec.conjugate().transpose())
        else:
            #This pays off for _N > 11
            self._Ut  = np.matmul(self.eig_vec, 
                            scipy_dia_matrix(np.exp(-1j*_n*self.eig_val*self.t)).multiply(self.eig_vec.conjugate().transpose()).toarray(), 
                            dtype = np.complex128)
        #possible further option?'
        #self._Ut = fastmat.matmul(self.eig_vec,np.diag(np.exp(-1j*self.eig_val*_t)))
        #t1 = timeit.default_timer()
        #print("Time Mat mult.: \t {}".format(t1-t0))

    def glue_circuit(self, axis: bool = 0, repetitions: int = 2):
        #General function to glue arbitrary GridQubit circuits 
        # given that they periodic boundary gates
        assert(repetitions > 1),\
            "AbstractModelError in glue_circuit: repetitions need to > 1, received {}".format(repetitions)
        if self.qubittype != "GridQubit":
            raise NotImplementedError()

        ###########################################################################
        #1.Find 2-qubit gates along the given axis
        # Through error if there are none
        # Use 2D list, [Moment][Operation]

        glueing_gates = [[] for _ in range(np.size(self.circuit.moments))]
        m = 0
        for moment in self.circuit.moments:
            #print("{}.Moment: \n{}\n ".format(i, moment.__dict__))
            for operation in moment._operations:
                #Here we are only interested in 2-qubit gates
                if len(operation._qubits) == 2:
                    if axis == 0:
                        #_row must be 0 and n0-1
                        #_col must be equal
                        if  ((operation._qubits[0]._row == 0 and operation._qubits[1]._row == self.n[0]-1 ) or \
                            (operation._qubits[0]._row == self.n[0]-1 and operation._qubits[1]._row == 0 )) and \
                            (operation._qubits[0]._col == operation._qubits[1]._col ) and self.n[0]>1:
                            #print("\nGlueing qubits: \t {}".format(operation._qubits))
                            glueing_gates[m].append(operation)
                    else:
                        #axis == 1
                        #_row must be equal
                        #_col must be n1-1
                        if  ((operation._qubits[0]._col == 0 and operation._qubits[1]._col == self.n[1]-1 ) or \
                            (operation._qubits[0]._col == self.n[1]-1 and operation._qubits[1]._col == 0 )) and \
                            (operation._qubits[0]._row == operation._qubits[1]._row )and self.n[1]>1:
                            #print("\nGlueing qubits: \t {}".format(operation._qubits))
                            glueing_gates[m].append(operation)
            m +=1
        #print("\nGlueing_gates: \t {}\n".format(glueing_gates))
        assert(any(glueing_gates)),\
            "AbstractModelError in glue_circuit: No periodic boundary 2-qubit gate found along glueing axis {}".format(int(axis))

        ###########################################################################
        #2. Generate glued circuit moment by moment
        #2a init non-glue Moment and its duplicates
        #       This includes duplicating/renaming sympy.Symbols
        #       This also includes dublicating/copying circuit_param_values
        #       Probably best done operation by operation in each Moment
        #2b. Add 2-Qubit Glueing gates to each moment 
        #2c. Calc/Determine new circuit_param/circuit_param_values based on existing
        ###########################################################################
        new_circuit = cirq.Circuit()

        #2a + 2b
        m = 0
        for moment in self.circuit.moments:
            #2a
            for operation in moment._operations:
                if operation not in glueing_gates[m]:
                    for i in range(repetitions):
                        #Add _g$i to any sympy.Symbol name
                        #Change if require qubit positions
                        new_circuit.append(self._get_operation_for_gc(operation, axis, i, "_g" + str(i)))
            #2b
            for operation in glueing_gates[m]:
                for i in range(repetitions):
                    new_circuit.append(self._get_operation_for_gc(operation, axis, i, "_g" + str(i), True, repetitions))
            m += 1
        #print(new_circuit)

        #2c
        new_cricuit_param =         []
        for i in range(repetitions):
            for j in range(len(self.circuit_param)):
                new_cricuit_param.append(sympy.Symbol(self.circuit_param[j].name + "_g" + str(i)))
        
        ###########################################################################
        #   3. overwrite qubits, circuit_param, circuit_param_values
        #   
        #   Overwrite is okay if previous copy works
        #       ISSUE: so far deepcopy(isng) does not work
        #   Need to overright this locally further to double further parameters locally!!
        ###########################################################################
        #Reset n
        self.n = self.n + (repetitions-1)*[self.n[0]*(1-axis) ,self.n[1]*axis]

        #Reset qubits    
        self.init_qubits(self.qubittype,self.n)

        #Reset cricuit_param 
        self.circuit_param = new_cricuit_param

        #Reset circuit_param_values
        self.circuit_param_values= np.tile(self.circuit_param_values, repetitions)

        #Reset circuit
        self.circuit= new_circuit

    def _get_operation_for_gc(  self,
                                operation: cirq.Operation, 
                                axis: bool, 
                                i: int, 
                                add_string: str,
                                glueing: bool = False,
                                repetitions: int = -1):
        _gate = operation._gate.__class__
        _gate_params = operation._gate.__dict__.copy()
        #print(_gate_params)
        #print(_gate_params.keys())
        _qubits = operation._qubits

        for key in list(_gate_params.keys()):
            if 'cached' in key or '_doc_' in key:
                # Remove dummies 
                _gate_params.pop(key)
                #print("Popped key: \t{}".format(key))
            else:
                if not isinstance(_gate_params.get(key), Number):
                    if isinstance(_gate_params.get(key), sympy.core.symbol.Symbol):
                        #print("key: \t{}\ntype: \t {}".format(key, type(_gate_params.get(key))))
                        _gate_params[key[1:]] = sympy.Symbol(_gate_params.pop(key).name + add_string)
                    else:
                        #type here: lass 'sympy.core.mul.Mul'
                        #print("key: \t{}\ntype: \t {}\ndict: \t {}"\
                        #    .format(key, type(_gate_params.get(key)), _gate_params.get(key).as_coeff_Mul()))
                        temp = _gate_params.pop(key).as_coeff_Mul()
                        _gate_params[key[1:]] = temp[0]*sympy.Symbol(temp[1].name + add_string)
                else:
                    _gate_params[key[1:]] = _gate_params.pop(key)

        
        if not glueing:
            #if i > 0 adapt _qubits
            if i > 0:
                _qubits = list(_qubits)
                for l in range(len(_qubits)):
                    _qubits[l]=cirq.GridQubit(  _qubits[l]._row + self.n[0]*i*(1-axis), 
                                                _qubits[l]._col+ self.n[1]*i*axis)
        else:
            #Always need to adapt qubits for glueing gates
            #Here we know that every gate is a 2 qubit gate -> _qubits[1] exists
            _qubits = list(_qubits)
            if i == 0: 
                _qubits[1]=cirq.GridQubit(  _qubits[1]._row + self.n[0]*(1-axis), 
                                            _qubits[1]._col+ self.n[1]*axis)
            else:
                for l in range(len(_qubits)):
                    _qubits[l]=cirq.GridQubit(  (_qubits[l]._row + self.n[0]*(i+l)*(1-axis))%(repetitions*self.n[0]), 
                                                (_qubits[l]._col+ self.n[1]*(i+l)*axis)%(repetitions*self.n[1]))

        yield _gate(**_gate_params).on(*_qubits)