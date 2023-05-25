import cirq
import collections
from itertools import chain
from numbers import Real
import numpy as np
import scipy
from sys import maxsize as sys_maxsize
from typing import Iterable,List, Union

# Order functions by Alphabet
# Rethink some of the nameing here...
def flatten(a) -> Iterable:
    """This function takes in a list of list or another nested iterable object and flattens it
    Args:
        a (Iterable): The nested iterable to be flattened
    Returns:
        Iterable: THe flattened iterable
    """
    for ii in a:
        # avoid strings and bytes to be split too
        if isinstance(ii,Iterable) and not isinstance(ii, (str, bytes)):
            yield from flatten(ii)
        else:
            yield ii

def get_gate_count(circuit: cirq.Circuit) -> int:
    """
        Note that this also counts cirq.reset gates
    """
    count = 0
    for moment in circuit.moments:
        count += len(moment.operations)
    return count

def greedy_grouping(paulisum: cirq.PauliSum) -> List[cirq.PauliSum]:
    # Missing doc string
    grouped_paulisums = []
    grouped_qubits = []

    #print(paulisum.__dict__)
    for paulistring, coefficient in paulisum._linear_dict.items():
        #print(paulistring)
        #print(coefficient)
        _tmp_qubits = []
        _tmp_paulisum = 1 #coefficient.copy()
        for item in paulistring:
            _tmp_qubits.append(item[0])
            _tmp_paulisum = _tmp_paulisum * item[1](item[0])

        _tmp_paulisum = cirq.PauliSum.from_pauli_strings(coefficient*_tmp_paulisum)
        _appended = False

        for i in range(len(grouped_qubits)):
            if _appended:
                continue
            if set(_tmp_qubits) & set(grouped_qubits[i]):
                pass
            else:
                for qubit in _tmp_qubits:
                    grouped_qubits[i].append(qubit)
                grouped_paulisums[i] += _tmp_paulisum
                _appended = True

        if not _appended:
            grouped_qubits.append(_tmp_qubits)
            grouped_paulisums.append(_tmp_paulisum)
    
    assert (sum(grouped_paulisums) == paulisum), "Error in greedy_grouping: paulisum:\n{}\nsum(grouped_paulis):\n{}\ngrouped_paulis:\n{}".format(
                paulisum, sum(grouped_paulisums), grouped_paulisums
            )
    return grouped_paulisums

def merge_same_gates(circuit: cirq.Circuit) -> cirq.Circuit:
    """
    Strategy:
        -Start from last moment and check whether operation commutes with operation on previous moment or is same operation
        -If same operation: merge angles
        -If commuting then check next moment
        -Do as long as moment of current operation > 1

    Resources:
        https://quantumai.google/cirq/transform/custom_transformers?authuser=4
        https://quantumai.google/cirq/start/intro?authuser=4
        https://quantumcomputing.stackexchange.com/questions/13488/reordering-commuting-gates-in-cirq-to-reduce-circuit-depth
        https://quantumai.google/reference/python/cirq/Circuit
    
    TODO:   Merge similar gates with different name e.g. X, Z into PhasedXZ?
            Very possibly exists already within Cirq.
            Remove identity gates
    """
    new_circuit = circuit.copy()
    #for i_moment in range(len(new_circuit)-1,-1,-1):
    for i_moment in range(len(new_circuit)-1,-1,-1):
        for i_op in range(len(new_circuit.moments[i_moment].operations)):
            current_op = new_circuit.moments[i_moment].operations[i_op]
            # find merge able operation
            for i_previous_moment in range(i_moment-1,-1,-1):
                for previous_op in new_circuit.moments[i_previous_moment]:
                    if set(current_op.qubits) == set(previous_op.qubits):
                        #Check first whether gates are the same 
                        # This potentially can be extended to more general by using the cirq mergers
                        # and extending them to operations with are neighbouring by vanishing commutator pathes
                        # print("Current op: {} \tPrevious op: {}\n".format(type(current_op.gate), previous_op.gate.__dict__))
                        if type(current_op.gate) == type(previous_op.gate):
                            # now check whether all operations between them commute
                            # We need to get all gates between both gates and check whether those commute with the gate
                            #print("Moment: {}\tCurrent op: {}\nMoment: {}\tPrevious op: {}\n"\
                            #    .format(i_moment, current_op.__dict__,i_previous_moment, previous_op.__dict__))
                            _IsMergable = True
                            for i_inter_moment in range(i_moment-1,i_previous_moment,-1):
                                if not _IsMergable:
                                    continue
                                #print("Moment: {}".format(i_inter_moment))
                                for inter_op in new_circuit.moments[i_inter_moment]:
                                    if not _IsMergable:
                                        continue
                                    if set(current_op.qubits) & set(inter_op.qubits):
                                        #print("Inter op: {}".format(inter_op.__dict__))
                                        #print(cirq.commutes(current_op.gate, inter_op.gate))
                                        #_IsMergable = cirq.definitely_commutes(current_op.gate, inter_op.gate)
                                        try:
                                            #print("Gate1: {}\tGate2: {}".format(type(current_op.gate), type(inter_op.gate)))
                                            _IsMergable = cirq.commutes(current_op.gate, inter_op.gate)
                                        except:
                                            #print("cirq.commutes fails:")
                                            #print("Gate1: {}\tGate2: {}".format(type(current_op.gate), type(inter_op.gate)))
                                            #print("Current_op: {}\tInter_op: {}".format(current_op, inter_op))
                                            #print("cirq.commutes fails:")
                                            #print("Moment: {}\tCurrent op: {}\nMoment: {}\tInter op: {}\nMoment: {}\tPrevious op: {}\n"\
                                            #     .format(i_moment, current_op.__dict__,i_inter_moment, inter_op.__dict__, i_previous_moment, previous_op.__dict__))

                                            # As cirq.commutes fails, Compare unitaries instead
                                            # Here we ensured already that the operations act on common qubits:
                                            _IsMergable = cirq.equal_up_to_global_phase(cirq.unitary(cirq.Circuit(current_op, inter_op) ),
                                                                                        cirq.unitary(cirq.Circuit(inter_op, current_op) ),
                                                                                        atol = 1e-10)

                            if _IsMergable:
                                # print("Mergeable operations: ")
                                # print("Moment: {}\tCurrent op: {}\nMoment: {}\tPrevious op: {}\n"\
                                #    .format(i_moment, current_op.__dict__,i_previous_moment, previous_op.__dict__))
                                #print(current_op.gate.__dict__)
                                #print(previous_op.gate.__dict__)

                                new_gate = current_op.gate.__class__()

                                for key, value in previous_op.gate.__dict__.items() :
                                    #print(current_op.gate.__dict__.get(key))
                                    value2=current_op.gate.__dict__.get(key)
                                    if value is not None and value2 is not None:
                                        setattr(new_gate, key, value+value2)
                                    else:
                                        setattr(new_gate, key, None)
                                #print(new_gate)

                                new_operation = new_gate.on(*current_op.qubits)
                                new_circuit.batch_replace([[i_moment,current_op, new_operation]])
                                current_op = new_operation
                                new_circuit.batch_remove([[i_previous_moment, previous_op]])
                                
                                #new_gate = current_op.gate + previous_op.gate
                                #new_circuit.insert(i_moment,new_gate.on(current_op.qubits))
    
    new_circuit = cirq.drop_empty_moments(cirq.drop_negligible_operations(new_circuit, atol=1e-15))
    
    # This is a bad assert as it uses cirq.unitary(circuit) somewhere
    #cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
    #    circuit, new_circuit
    #)
    return new_circuit

def alternating_indices_to_sectors(M,even_first: bool = True) -> np.ndarray:
    """
        This function takes a matrix and reorders so that the even index basis vectors 
        are put at the beginning and the odd are put at the end. Mostly useful for
        openfermion stuff, as the matrices usually set up alternating rows of up and 
        down spins vectors, and not sectors. i.e. this reorganizes the indices from
        
        u11 d11 u12 d12 u21 d21 u22 d22
        to
        u11 u12 u21 u22 d11 d12 d21 d22

        Args:
            M (np.ndarray): the matrix to be reordered
            even_first: whether the even vectors go in the first sector or the last (0 is the first index)
    """
    if not isinstance(M,np.ndarray):
        M=np.array(M)
    idxs = (np.array(list(chain(    range(np.mod(int(even_first)+1,2),ii,2),
                                    range(int(even_first),ii,2)))) for ii in M.shape)
    return M[np.ix_(*idxs)]

def flip_cross_rows(M,
                    flip_odd=True):
    """
        Reverses the order of the elements in odd or even rows.
        
        Args:
            M (n-by-m matrix): Input matrix
            flip_odd (bool, optional): Whether to reverse the odd or even rows. Defaults to True (odd rows)     .
        
        Returns:
            M: New matrix with odd or even rows flipped
    """
    M_tmp = np.array(M)
    M_tmp[int(flip_odd)::2,:] = M_tmp[int(flip_odd)::2,::-1]
    return M_tmp

def hamming_weight(binary: Union[int,str])-> int:
    """Counts the number of 1s in a binary number. Can input either a binary or int representation
    Args:
        n (str i.e. binary representation or int representation): the number to be processed
    Returns:
        int: the hamming weight, i.e. the number of 1s in the number n
    """
    if isinstance(binary,int):
        binary=bin(binary)
    elif isinstance(binary,str):
        pass
    else:
        raise TypeError("expected a binary number or an int but got a {}".format(type(binary)))
    return sum((1 for j in binary if j == '1'))

def index_bits(binary: Union[int,str],ones=True) -> list:
    """Takes a binary number and returns a list of indices where the bit is one (or zero)
    Args:
        a (binary number): The binary number whose ones or zeroes will be indexed
        ones (bool): If true, index ones. If false, index zeroes
    Returns:
        list: List of indices where a is one (or zero)
    """
    if isinstance(binary,int):
        binary = bin(binary)
        
    b = binary.split("b")[1]

    if ones:
        return [idx for idx, v in enumerate(b) if int(v)]
    else:
        return [idx for idx, v in enumerate(b) if not int(v)]

def interweave(a, b)-> np.ndarray:
    """This function interweaves to arrays, creating an array c whose even indices contain a's items and odd indices contain b's items
    Args:
        a (np.ndarray): the even-index array
        b (np.ndarray): the odd-index array
    Returns:
        c (np.ndarray): the array maded of interwoven a and b
    """
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c

def direct_sum(a,b):
    """
        Missing docstring
    """
    return np.block(    [[a,np.zeros((a.shape[0],b.shape[1]))],
                        [np.zeros((b.shape[0],a.shape[1])),b]])

def generalized_matmul(multiplication_rule = np.matmul,
                          *args):
    """
        This function takes in multiple matrices (as different arguments) and multiplies them together 
        via the multiplication rule function. Other functions can be np.kron or fauvqe.direct_sum
        
        Returns:
            Matrix: The matrix resulting from the multiplication
    """
    R=multiplication_rule(args[0],args[1])
    if len(args)>2:
        for M in args[2:]:
            R=multiplication_rule(R,M)
    return R

#This seems somewhat specific to a fermion problem?
def sectors_to_alternating_indices(M, 
                                   even_first: bool = True, 
                                   axis=None) -> np.ndarray:
    """
    This function turns a matrix which has two "sectors" into an interwoven one
    i.e.

    u u d d
    a a b b
    x x y y
    c c d d

    into

    u d u d
    x y x y
    a b a b
    c d c d

    Args:
        M (np.ndarray): the matrix to be reordered
        even_first (bool, optional): whether the first sector is distributed over even indices (0 is the first index). Defaults to True.

    Returns:
        M (np.ndarray): reordered matrix
    """
    M = np.array(M)
    dims = M.shape
    idxs = []
    if isinstance(axis, int):
        axis = (axis,)
    for ind, dim in enumerate(dims):
        half1, half2 = np.arange(0, np.floor(dim / 2)), np.arange(
            np.floor(dim / 2), dim
        )
        if not even_first:
            half1, half2 = half2, half1
        if axis is None:
            idxs.append(np.array(interweave(half1, half2)).astype(int))
        elif ind in axis:
            idxs.append(np.array(interweave(half1, half2)).astype(int))
    return M[np.ix_(*idxs)]
############################################################################################
#                                                                                          #
#                    Utilities for Quantum Mechanics                                       #
#                                                                                          #
############################################################################################
def commutator(A: np.array, B:np.array) -> np.array:
    """
        Commutator of A and B
        
        Parameters
        ----------
        self
        A: np.array
            Matrix 1
        B: np.array
            Matrix 2
        
        Returns
        -------
        [A, B]
    """
    # Not that this might be misleading as there is a commutator of
    # cirq.PauliSum from openfermion which we want to use as well
    # possible solution allow both
    return A @ B - B @ A

def orth_norm(A: np.array) -> Real:
    """
        TE:     THis is badly written and super inefficent. just calculated largest and smallest eigenvalue.
                Due to symmetry often it is sufficent to just calculate one of them

        Calculates the orthogonal norm of A
        
        Parameters
        ----------
        self
        A: np.array
            matrix of which orthogonal norm is calculated
        
        Returns
        -------
        ||A||_\perp
    """
    eig_val = scipy.linalg.eigvalsh(A)
    return (eig_val[-1] - eig_val[0])/2

def ptrace(Q, ind):
    """
        Calculates partial trace of A over the indices indicated by ind. 
        Major parts of this function are copied from qutip 4.7's _ptrace_dense function.
        
        Parameters
        ----------
        self
        A: np.array
            matrix which is partially traced over
        ind: List[np.uint]
            indices which are being traced 
        
        Returns
        -------
        Tr_ind(A): np.array
    """
    n = np.log2(len(Q))
    assert abs(n - int(n)) < 1e-13, "Wrong matrix size. Required 2^n, Received {}".format(n)
    n = int(n)
    
    if isinstance(ind, int):
        ind = np.array([ind])
    else:
        ind = np.asarray(ind)
    for x in ind:
        if not 0 <= x < n:
            raise IndexError("Invalid selection index in ptrace.")
    sel = np.asarray([i for i in range(n) if i not in ind])
    rd = np.asarray([2]*n, dtype=np.int32).ravel()
    sel = list(np.sort(sel))
    dkeep = (rd[sel]).tolist()
    qtrace = list(set(np.arange(n)) - set(sel))
    dtrace = (rd[qtrace]).tolist()
    if not dtrace:
        # If we are keeping all dimensions, no need to construct an ndarray.
        return Q.copy()
    rd = list(rd)
    return np.trace(Q
                    .reshape(rd + rd)
                    .transpose(qtrace + [n + q for q in qtrace] +
                            sel + [n + q for q in sel])
                    .reshape([np.prod(dtrace, dtype=np.int32),
                            np.prod(dtrace, dtype=np.int32),
                            np.prod(dkeep, dtype=np.int32),
                            np.prod(dkeep, dtype=np.int32)]))

