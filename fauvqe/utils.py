from typing import Iterable,Union
import numpy as np
from itertools import chain
import collections

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
    for ii in a:
        # avoid strings and bytes to be split too
        if isinstance(ii,Iterable) and not isinstance(ii, (str, bytes)):
            yield from flatten(ii)
        else:
            yield ii

def pi_matmul(*args):
	R=np.matmul(args[0],args[1])
	if len(args)>2:
		for M in args[2:]:
			
			R=np.matmul(R,M)
	return R

def print_non_zero(M,name: str=None, eq_tol: float=1E-15):
	if name is not None:
		print(name)
	print((abs(M)>eq_tol).astype(int))


def alternating_indices_to_sectors(M,even_first: bool = True):
    """This function takes a matrix and reorders so that the even index basis vectors 
    are put at the beginning and the odd are put at the end. Mostly useful for
    openfermion stuff, as the matrices usually set up alternating rows of up and 
    down spins vectors, and not sectors. i.e. this reorganizes the indices from

    u11 d11 u12 d12 u21 d21 u22 d22

    to

    u11 u12 u21 u22 d11 d12 d21 d22

    Args:
        M (np.array): the matrix to be reordered
        even_first: whether the even vectors go in the first sector or the last
    """
    M_tmp = M.copy()
    dims = M.shape
    if even_first:
        a=0
        b=1
    else:
        a=1
        b=0
    idxs = (np.array(list(chain(range(a,ii,2),range(b,ii,2)))) for ii in dims)
    return M[np.ix_(*idxs)]
    
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
