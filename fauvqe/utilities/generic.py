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

#Needs to be tested:
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

        _tmp_paulisum = coefficient*_tmp_paulisum
        _appended = False

        for i in range(len(grouped_qubits)):
            if set(_tmp_qubits) & set(grouped_qubits[i]):
                pass
            else:
                for qubit in _tmp_qubits:
                    grouped_qubits[i].append(qubit)
                grouped_paulisums[i] += _tmp_paulisum
                _appended = True
                continue

        if not _appended:
            grouped_qubits.append(_tmp_qubits)
            grouped_paulisums.append(_tmp_paulisum)
    
    assert (sum(grouped_paulisums) == paulisum), "Error in greedy_grouping: paulisum:\n{}\nsum(grouped_paulis):\n{}".format(
                paulisum, sum(grouped_paulisums)
            )
    return grouped_paulisums

def merge_same_gates(circuit: cirq.Circuit) -> cirq.Circuit:
    # strategy:
    # start from last moment see if operation commuts with operation on previous moment
    # or is same operation
    # If same operation: merge angles
    # if commuting then check next moment
    # do until as long as moment of current operation > 1
    # Does this help?
    #https://quantumai.google/cirq/transform/custom_transformers?authuser=4
    #https://quantumai.google/cirq/start/intro?authuser=4
    #https://quantumcomputing.stackexchange.com/questions/13488/reordering-commuting-gates-in-cirq-to-reduce-circuit-depth
    #https://quantumai.google/reference/python/cirq/Circuit
    new_circuit = circuit.copy()
    for i_moment in range(len(circuit)-1,-1,-1):
        for i_op in range(len(circuit.moments[i_moment].operations)):
            current_op = circuit.moments[i_moment].operations[i_op]
            # find merge able operation
            for i_previous_moment in range(i_moment-1,-1,-1):
                for previous_op in circuit.moments[i_previous_moment]:
                    if current_op.qubits == previous_op.qubits:
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
                            for i_inter_moments in range(i_moment-1,i_previous_moment,-1):
                                if not _IsMergable:
                                    continue
                                #print("Moment: {}".format(i_inter_moments))
                                for inter_op in circuit.moments[i_inter_moments]:
                                    if not _IsMergable:
                                        continue
                                    if set(current_op.qubits) & set(inter_op.qubits):
                                        #print("Inter op: {}".format(inter_op.__dict__))
                                        #print(cirq.commutes(current_op.gate, inter_op.gate))
                                        _IsMergable = cirq.definitely_commutes(current_op.gate, inter_op.gate)
                            if _IsMergable:
                                print("Mergeable operations: ")
                                print("Moment: {}\tCurrent op: {}\nMoment: {}\tPrevious op: {}\n"\
                                    .format(i_moment, current_op.__dict__,i_previous_moment, previous_op.__dict__))
                                print(current_op.gate.__dict__)
                                print(previous_op.gate.__dict__)

                                new_gate = current_op.gate.__class__()

                                for key, value in previous_op.gate.__dict__.items() :
                                    #print(current_op.gate.__dict__.get(key))
                                    value2=current_op.gate.__dict__.get(key)
                                    if value is not None and value2 is not None:
                                        setattr(new_gate, key, value+value2)
                                    else:
                                        setattr(new_gate, key, None)
                                print(new_gate)

                                new_operation = new_gate.on(*current_op.qubits)
                                new_circuit.batch_replace([[i_moment,current_op, new_operation]])
                                new_circuit.batch_remove([[i_previous_moment, previous_op]])
                                
                                #new_gate = current_op.gate + previous_op.gate
                                #new_circuit.insert(i_moment,new_gate.on(current_op.qubits))
    new_circuit = cirq.drop_empty_moments(new_circuit)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, new_circuit
    )
    return new_circuit

def alternating_indices_to_sectors(M,even_first: bool = True) -> np.ndarray:
    """This function takes a matrix and reorders so that the even index basis vectors 
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
    M=check_type_and_convert(M)
    dims = M.shape
    if even_first:
        a=0
        b=1
    else:
        a=1
        b=0
    idxs = (np.array(list(chain(range(a,ii,2),range(b,ii,2)))) for ii in dims)
    return M[np.ix_(*idxs)]

def check_type_and_convert(M):
    """
        Missing docstring
    """
    if not isinstance(M,np.ndarray):
        return np.array(M)
    return M

def direct_sum(a,b):
    """
        Missing docstring
    """
    ax=a.shape[0]
    ay=a.shape[1]
    
    bx=b.shape[0]
    by=b.shape[1]
    return np.block(    [[a,np.zeros((ax,by))],
                        [np.zeros((bx,ay)),b]])

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

def flatten_qubits(gridqubits):
    """
        Missing docstring
    """
    return list(flatten(gridqubits))

def flip_cross_rows(M,flip_odd=True):
    """Reverses the order of the elements in odd or even rows.
    Args:
        M (n-by-m matrix): Input matrix
        flip_odd (bool, optional): Whether to reverse the odd or even rows. Defaults to True (odd rows)     .
    Returns:
        M: New matrix with odd or even rows flipped
    """
    M_tmp = np.array(M)
    if flip_odd==True:
        a=1
    else:
        a=0
    M_tmp[a::2,:] = M_tmp[a::2,::-1]
    return M_tmp

def hamming_weight(n: Union[int,str])-> int:
    """Counts the number of 1s in a binary number. Can input either a binary or int representation
    Args:
        n (str i.e. binary representation or int representation): the number to be processed
    Returns:
        int: the hamming weight, i.e. the number of 1s in the number n
    """
    if isinstance(n,int):
        n=bin(n)
    elif isinstance(n,str):
        pass
    else:
        raise TypeError("expected a binary number or an int but got a {}".format(type(n)))
    return sum((1 for j in n if j == '1'))

def index_bits(a: str,ones=True) -> list:
    """Takes a binary number and returns a list of indices where the bit is one (or zero)
    Args:
        a (binary number): The binary number whose ones or zeroes will be indexed
        ones (bool): If true, index ones. If false, index zeroes
    Returns:
        list: List of indices where a is one (or zero)
    """
    if isinstance(a,int):
        a = bin(a)
    b = a.split("b")[1]
    if ones:
        return [idx for idx, v in enumerate(b) if int(v)]
    elif not ones:
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

def lists_almost_have_same_elements(a: list,b: list,decimals:int):
    """
        Missing docstring
        Use np.isclose instead
    """
    rounded_a=np.round(np.array(a),decimals=decimals)
    rounded_b=np.round(np.array(b),decimals=decimals)
    return lists_have_same_elements(rounded_a,rounded_b)

def lists_have_same_elements(a: list,b: list):
    """
        Missing docstring
    """
    return collections.Counter(a) == collections.Counter(b)

def niceprint(a: np.array,precision: int=2, suppress: bool = True, threshold: int=sys_maxsize):
    """This function nicely prints a numpy array without truncation to desired precision
    Args:
        a (np.array): the array to print. Converted to np.array if not
        precision (int, optional): number of significant numbers. Defaults to 2.
        suppress (bool, optional): print very small numbers as 0 (small as 0 with the current precision) . Defaults to True.
        threshold (int, optional): The number of items to show. Defaults to sys.maxsize, i.e. as many as allowed.
    """
    if not isinstance(a,np.ndarray):
        a=np.array(a)
    with np.printoptions(precision=precision,suppress=suppress, threshold=threshold): 
        print(a)

def pi_direct_sum(*args):
    """
        Missing docstring
    """
    R=direct_sum(args[0],args[1])
    if len(args) > 2:
        for M in args[2:]:
            R=direct_sum(R,M)
    return R

def pi_kron(*args):
    """
        Missing docstring
    """
    R=np.kron(args[0],args[1])
    if len(args)>2:
        for M in args[2:]:
            R=np.kron(R,M)
    return R

def pi_matmul(*args):
    """This function takes in multiple matrices (as different arguments) and multiplies them together
    Returns:
        Matrix: The matrix resulting from the multiplication
    """
    R=np.matmul(args[0],args[1])
    if len(args)>2:
    	for M in args[2:]:
            R=np.matmul(R,M) 
    return R

def print_non_zero( M,
                    name: str=None, 
                    eq_tol: float=1E-15):
    if name is not None: 
        print(name)
    print((abs(M)>eq_tol).astype(int))

def round_small_to_zero(l: list,eq_tol: float = 1E-15):
    """
        Missing docstring
    """
    return [0 if abs(x) < eq_tol else x for x in l]

def sectors_to_alternating_indices(M,even_first: bool = True) -> np.ndarray:
    """This function turns a matrix which has two "sectors" into an interwoven one
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
    M=check_type_and_convert(M)
    dims = M.shape
    if even_first:
        idxs = (np.array(interweave(np.arange(0,np.floor(ii/2)),np.arange(np.floor(ii/2),ii))).astype(int) for ii in dims)
    else:
        idxs = (np.array(interweave(np.arange(np.floor(ii/2),ii),np.arange(0,np.floor(ii/2)))).astype(int) for ii in dims)
    return M[np.ix_(*idxs)]

def unitary_transpose(M):
    """
        Missing docstring
    """
    return np.conj(np.transpose(np.array(M)))

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
    #print((eig_val[-1] - eig_val[0])/2)
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

