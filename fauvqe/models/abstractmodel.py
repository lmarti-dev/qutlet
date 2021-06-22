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
import abc
from typing import Tuple, List
from numbers import Number, Real

import numpy as np
import sympy
import cirq
import qsimcirq
import timeit
import fastmat

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
        self.hamiltonian = cirq.PauliSum()
        self.init_qubits(qubittype, n)
        self.set_simulator()
        self.t : Real = 0
        

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
    def set_simulator(self, simulator_name="qsim", simulator_options: dict = {}):
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
            """
            self.simulator_options = {"t": 8, "f": 4}
            self.simulator_options.update(simulator_options)
            self.simulator = qsimcirq.QSimSimulator(self.simulator_options)
        elif simulator_name == "cirq":
            self.simulator_options = {}
            self.simulator = cirq.Simulator()
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

    def Ut(self, t: Number):
        #https://docs.sympy.org/latest/modules/numeric-computation.html
        #1.If exact diagonalisation exists already, don't calculate it again
        # Potentially rather use scipy here!
        _N = 2**(np.size(self.qubits))
        try:
            if np.size(self.eig_val) != _N or \
            (np.shape(self.eig_vec) != np.array((_N, _N)) ).all():
                self.diagonalise(solver = "scipy", solver_options={"subset_by_index": [0, _N - 1]})
        except:
            # It might not work if self.eig_val does not exist yet
            self.diagonalise(solver = "scipy", solver_options={"subset_by_index": [0, _N - 1]})
        
        #print("eig_val: \t {}, eig_vec \t {}, _N \t {}".\
        #                format(np.size(self.eig_val), np.shape(self.eig_vec) ,_N))
        
        #2. The do exact diagonlisation + use sympy symbol
        # U(t) = (v1 .. vN) diag(e⁻iE1t .. e-iENt) (v1 .. vN)^+
        t0 = timeit.default_timer()
        if np.size(self.qubits) < 12:
            self._Ut = np.matmul(np.matmul(self.eig_vec,np.diag(np.exp(-1j*self.eig_val*t)), dtype = np.complex64),
                            self.eig_vec.conjugate())
        else:
            #This pays off for _N > 11
            self._Ut  = np.matmul(self.eig_vec, 
                            scipy_dia_matrix(np.exp(-1j*self.eig_val*t)).multiply(self.eig_vec.conjugate()).toarray(), 
                            dtype = np.complex64)
        #possible further option?'
        #self._Ut = fastmat.matmul(self.eig_vec,np.diag(np.exp(-1j*self.eig_val*_t)))
        t1 = timeit.default_timer()
        print("Time Mat mult.: \t {}".format(t1-t0))

        #return self._Ut