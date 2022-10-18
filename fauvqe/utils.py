from typing import Iterable,Union
import numpy as np
from itertools import chain
import collections
import sys 
from datetime import datetime
import os
import io
import re
import json

def now_str():
    return datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

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

def print_non_zero(M,name: str=None,eq_tol: float=1E-15): # pragma: no cover
    if name is not None:
        print(name)
    print(get_non_zero(M=M,eq_tol=eq_tol))

def get_non_zero(M,eq_tol: float=1E-15):
	return (abs(np.array(M))>eq_tol).astype(int)

def alternating_indices_to_sectors(M,even_first: bool = True,axis=None) -> np.ndarray:
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
    M=np.array(M)
    dims = M.shape
    if even_first:
        a=0
        b=1
    else:
        a=1
        b=0
    if axis is None:
        idxs = (np.array(list(chain(range(a,ii,2),range(b,ii,2)))) for ii in dims)
    else:
        idxs = (np.array(list(chain(range(a,ii,2),range(b,ii,2)))) if axis==ind else np.arange(ii) for ind,ii in enumerate(dims))
    return M[np.ix_(*idxs)]

def interweave(a, b)-> np.ndarray:
    """This function interweaves to arrays, creating an array c whose even indices contain a's items and odd indices contain b's items.
    When one array is shorter than the other, the function will simply keep using the longer array's items, i.e. stop interweaving

    Args:
        a (np.ndarray): the even-index array
        b (np.ndarray): the odd-index array

    Returns:
        c (np.ndarray): the array made of interwoven a and b
    """
    
    c=[]
    for i in range(max(len(a),len(b))):
        if i < len(a):
            c.append(a[i])
        if i < len(b):
            c.append(b[i])
    return c

def sectors_to_alternating_indices(M,even_first: bool = True,axis=None) -> np.ndarray:
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
    M=np.array(M)
    dims = M.shape
    idxs=[]
    for ii in dims:
        half1,half2=np.arange(0,np.floor(ii/2)),np.arange(np.floor(ii/2),ii)
        if not even_first:
            half1,half2=half2,half1
        if axis is None:
            idxs = (np.array(interweave(half1,half2)).astype(int) for ii in dims)
        else:
            idxs = (np.array(interweave(half1,half2)).astype(int) if axis==ind else np.arange(ii) for ind,ii in enumerate(dims))
    return M[np.ix_(*idxs)]

def flip_cross(M,rc="r",flip_odd=True):
    if rc=="r":
        return flip_cross_rows(M=M,flip_odd=flip_odd)
    if rc=="c":
        return flip_cross_cols(M=M,flip_odd=flip_odd)
    else:
        raise ValueError("Expected rc to be r or c, got: {}".format(rc))
def flip_cross_cols(M,flip_odd=True):
    M_tmp = np.array(M)
    if flip_odd==True:
        a=1
    else:
        a=0
    M_tmp[:,a::2] = M_tmp[::-1,a::2]
    if isinstance(M,list):
        return M_tmp.tolist()
    return M_tmp
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
    if isinstance(M,list):
        return M_tmp.tolist()
    return M_tmp

def lists_have_same_elements(a: Iterable,b: Iterable):
    return collections.Counter(a) == collections.Counter(b)


## Use np.isclose instead
# def lists_almost_have_same_elements(a: list,b: list,decimals:int):
#     rounded_a=np.round(np.array(a),decimals=decimals)
#     rounded_b=np.round(np.array(b),decimals=decimals)
#     return lists_have_same_elements(rounded_a,rounded_b)

def round_small_to_zero(l: list,eq_tol: float = 1E-15):
    zl=[0 if abs(x) < eq_tol else x for x in l]
    return zl

def flatten_qubits(gridqubits): # pragma: no cover
    return list(flatten(gridqubits))

def niceprint(a: np.array,precision: int=2, suppress: bool = True, threshold: int=sys.maxsize): #pragma: no cover
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
        if bin(int(n,2)) != n:
            raise TypeError("Expected a valid binary number string, but got {}".format(n))
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

def arg_alternating_index_to_sector(index: int,N: int):
    """This takes in an index and length of the array and returns the index of
    the equivalent sectorized matrix. The argsort equivalent to alternating_indices_to_sectors

    Args:
        index (int): the index to be sectorized
        N (tuple): the vector length

    Returns:
        int: The sectorized index
    """
    return int((np.ceil(N/2).astype(int)-1 + (index+1)/2)*(index%2) + (1-index%2)*(index/2))

def arg_alternating_indices_to_sectors(indices: tuple,N: Union[tuple,int]):
    if isinstance(N,tuple): 
        if len(N) != len(indices):
            raise TypeError("The length of N is not equal to the length of the indices vector")
        return tuple(map(arg_alternating_index_to_sector,indices,N))
    elif isinstance(N,int):
        return tuple(map(arg_alternating_index_to_sector,indices,[N]*len(indices)))
    else:
        raise TypeError("Expected N to be either a tuple or an int, got a {}".format(type(N)))

def arg_flip_cross_row(x:int,y:int,dimy:int,flip_odd:bool=True):
    """The arg equivalent of flip cross rows

    Args:
        x (int): the x index of the matrix
        y (int): the y index of the matrix
        dimy (int): the y dimension of the matrix
        flip_odd (bool, optional): Whether to flip the odd or even row indices. Defaults to True.
    """ 
    if x < 0 or y < 0 or y >= dimy or dimy <= 0:
        raise ValueError("Expected positives indices and dimension, got x:{x},y:{y},dimy:{dimy}".format(x=x,y=y,dimy=dimy)) 
    if flip_odd:
        a=1
        b=0
    else:
        a=0
        b=1
    if x%2 == a:
        return x,dimy-1-y
    elif x%2 == b:
        return x,y

def grid_to_linear(x,y,dimx,dimy,horizontal=True):
    if horizontal:
        return x*dimy+y
    else:
        return y*dimx+x

def linear_to_grid(n,dimx,dimy,horizontal=True):
    if horizontal:
        return np.unravel_index((n),(dimx,dimy),order="C")
    else:
        return np.unravel_index((n),(dimx,dimy),order="F")


def pretty_convert_list(l):
    ll=[]
    for x in l:
        ll.append(type_to_str(v=x,max_size=4))
    return ll

def type_to_str(v,max_size=4) -> str:
    out_v=None
    if isinstance(v,list):
        if len(v) > max_size:
            out_v = str(len(v))
        else:
            out_v = pretty_convert_list(v)
    elif isinstance(v,np.ndarray):
        if len(v.shape)==1:
            if len(v)> max_size:
                out_v = "list:"+str(len(v))
            else:
                out_v = pretty_convert_list(v)
        else:
            out_v = "ndarray:"+str(v.shape)
    elif any(isinstance(v,t) for t in (str,int)):
        out_v = str(v)
    elif any(isinstance(v,t) for t in(float,np.double,np.float64,np.single)):
        out_v = "{v:.3}".format(v=v)
    elif any(isinstance(v,t) for t in (np.csingle,np.cdouble)):
        out_v = "{r:.3}+j{i:.3}".format(r=np.real(v),i=np.imag(v))
    else:
        out_v = "unsupported type:{t}".format(t=type(v))
    
    return out_v


def infodump_locals(locals,max_size=4):
    infodump={}
    for k,v in locals.items():
        try:
            if isinstance(v,type(np)) or v is None or "__" in k:
                pass
            else:
                infodump[k]=type_to_str(v,max_size)
        except ValueError:
            infodump[k] = str(v)
    return infodump
    
def normalize(v: np.ndarray) -> np.ndarray:
    return v/np.linalg.norm(v)

def sum_divisible(l:list,i:int):
    return sum([1 if x%i==0 else 0 for x in l])
def sum_even(l):
    return sum_divisible(l,2)
def sum_odd(l):
    return len(l) - sum_even(l)


def ensure_fpath(fpath:os.PathLike):
    """If the file path doesn't exist, create the necessary directories

    Args:
        fpath (os.PathLike): the file path (with the file at the end)
    """
    dirname = os.path.dirname(fpath)
    if not os.path.exists(dirname): 
        os.makedirs(dirname)


def normalize_str(s: str):
    return re.sub(r"\W","",s)

def cap_first(s: str):
    if len(s)==1:
        return s[0].upper()
    return s[0].upper() + s[1:]

def random_name(lenw=5,Nwords=3):
    word_file = "/usr/share/dict/words"
    words = io.open(word_file,mode="r",encoding="utf8").read().splitlines()
    words = [normalize_str(w) for w in words if len(w)==lenw]
    indices = np.random.choice(len(words),Nwords,replace=True)
    return "".join([cap_first(words[iii]) for iii in indices])


def fidelity(a,b):
    if np.all(a.shape != b.shape):
        raise ValueError("vectors do not have the same shape a:{a},b:{b}".format(a=a.shape,b=b.shape))
    return np.sqrt(np.abs(np.dot(np.conj(a),b)*np.dot(np.conj(b),a)))
def infidelity(a,b):
    return 1-fidelity(a,b)

def save_to_json(data,fpath=None,randname=False):
    sobj=json.dumps(data,ensure_ascii=False,indent=4)
    if fpath is None:
        fpath = random_name() + "_" + now_str()
    elif randname:
        fpath = fpath + "_" + random_name() + "_" + now_str()
    else:
        fpath = fpath + "_" + now_str()
    
    ensure_fpath(fpath)
    fout=io.open("{fpath}.json".format(fpath=fpath),"w+",encoding="utf8")
    fout.write(sobj)
    fout.close()
    print("saved {}".format(fpath))