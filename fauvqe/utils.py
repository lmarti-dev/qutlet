from typing import Iterable,Union
import numpy as np
from itertools import chain
import collections
import sys 

def pi_kron(*args):
	R=np.kron(args[0],args[1])
	if len(args)>2:
		for M in args[2:]:
			
			R=np.kron(R,M)
	return R
	
def direct_sum(a,b):

	ax=a.shape[0]
	ay=a.shape[1]
	
	bx=b.shape[0]
	by=b.shape[1]
	
	R=np.block(
		[[a,np.zeros((ax,by))],
		[np.zeros((bx,ay)),b]])
	return R

def pi_direct_sum(*args):
	R=direct_sum(args[0],args[1])
	if len(args)>2:
		for M in args[2:]:
			R=direct_sum(R,M)
	return R

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

def print_non_zero(M,name: str=None, eq_tol: float=1E-15):
	if name is not None:
		print(name)
	print((abs(M)>eq_tol).astype(int))

def check_type_and_convert(M):
    if not isinstance(M,np.ndarray):
        return np.array(M)
    return M

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

def lists_have_same_elements(a: list,b: list):
    return collections.Counter(a) == collections.Counter(b)


## Use np.isclose instead
def lists_almost_have_same_elements(a: list,b: list,decimals:int):
    rounded_a=np.round(np.array(a),decimals=decimals)
    rounded_b=np.round(np.array(b),decimals=decimals)
    return lists_have_same_elements(rounded_a,rounded_b)

def round_small_to_zero(l: list,eq_tol: float = 1E-15):
    zl=[0 if abs(x) < eq_tol else x for x in l]
    return zl

def flatten_qubits(gridqubits):
    return list(flatten(gridqubits))

def niceprint(a: np.array,precision: int=2, suppress: bool = True, threshold: int=sys.maxsize):
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


def unitary_transpose(M):
    return np.conj(np.transpose(np.array(M)))
