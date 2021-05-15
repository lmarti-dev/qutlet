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

import numpy as np
import sympy
import cirq
import qsimcirq


class Initialiser(abc.ABC):
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
        self.circuit_param: List[sympy.Symbol] = []
        self.init_qubits(qubittype, n)
        self.circuit = cirq.Circuit()
        self.set_simulator()

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
            self.simulator_options = {"t": 8}
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
        raise NotImplementedError()
