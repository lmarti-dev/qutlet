from typing import Iterable, Union
import numpy as np
from scipy.linalg import sqrtm
from itertools import chain
import collections
import sys
from datetime import datetime
import os
import io
import re
import json

import fauvqe.json


def now_str() -> str:  # pragma: no cover
    """Returns the current date up to the second as a string under the format YYYY-MM-DD-hh-mm-ss

    Returns:
        str: the current date as a string
    """
    return datetime.today().strftime("%Y-%m-%d-%H-%M-%S")


def pi_kron(*args) -> np.ndarray:
    """Compute the tensor product of all matrices given as input

    Returns:
        np.ndarray: the tensor product of the matrices
    """
    R = np.kron(args[0], args[1])
    if len(args) > 2:
        for M in args[2:]:
            R = np.kron(R, M)
    return R


def direct_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes the direct sum between two matrices

    Args:
        a (np.ndarray): the first input matrix
        b (np.ndarray): the second input matrix

    Returns:
        np.ndarray: The direct sum of a and b
    """
    ax = a.shape[0]
    ay = a.shape[1]

    bx = b.shape[0]
    by = b.shape[1]

    R = np.block([[a, np.zeros((ax, by))], [np.zeros((bx, ay)), b]])
    return R


def pi_direct_sum(*args):
    """Compute the direct sum of all matrices given as arguments (i.e. creates a large matrix where all args are set as a block)

    Returns:
        np.ndarray: the output matrix whose diagonal is made of blocks of the input matrices
    """
    R = direct_sum(args[0], args[1])
    if len(args) > 2:
        for M in args[2:]:
            R = direct_sum(R, M)
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
        if isinstance(ii, Iterable) and not isinstance(ii, (str, bytes)):
            yield from flatten(ii)
        else:
            yield ii


def pi_matmul(*args) -> np.ndarray:
    """This function takes in multiple matrices (as different arguments) and multiplies them together
    Returns:
        Matrix: The matrix resulting from the multiplication
    """
    R = np.matmul(args[0], args[1])
    if len(args) > 2:
        for M in args[2:]:
            R = np.matmul(R, M)
    return R


def get_non_zero(M, eq_tol: float = 1e-15) -> np.ndarray:
    """Returns an array with ones where the input array is non-null and zero elsewhere

    Args:
        M (np.ndarray): the input array
        eq_tol (float, optional): the tolerance below which a number is considered to be 0. Defaults to 1e-15.

    Returns:
        np.ndarray: the output array with only filled with ones and zeroes.
    """
    return (abs(np.array(M)) > eq_tol).astype(int)


def alternating_indices_to_sectors(M, even_first: bool = True, axis=None) -> np.ndarray:
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
    M = np.array(M)
    dims = M.shape
    if even_first:
        a = 0
        b = 1
    else:
        a = 1
        b = 0
    if axis is None:
        idxs = (np.array(list(chain(range(a, ii, 2), range(b, ii, 2)))) for ii in dims)
    else:
        idxs = (
            np.array(list(chain(range(a, ii, 2), range(b, ii, 2))))
            if axis == ind
            else np.arange(ii)
            for ind, ii in enumerate(dims)
        )
    return M[np.ix_(*idxs)]


def interweave(a, b) -> np.ndarray:
    """This function interweaves to arrays, creating an array c whose even indices contain a's items and odd indices contain b's items.
    When one array is shorter than the other, the function will simply keep using the longer array's items, i.e. stop interweaving

    Args:
        a (np.ndarray): the even-index array
        b (np.ndarray): the odd-index array

    Returns:
        c (np.ndarray): the array made of interwoven a and b
    """

    c = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            c.append(a[i])
        if i < len(b):
            c.append(b[i])
    return c


def sectors_to_alternating_indices(M, even_first: bool = True, axis=None) -> np.ndarray:
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


def flip_cross(M, rc="r", flip_odd=True):
    if rc == "r":
        return flip_cross_rows(M=M, flip_odd=flip_odd)
    if rc == "c":
        return flip_cross_cols(M=M, flip_odd=flip_odd)
    else:
        raise ValueError("Expected rc to be r or c, got: {}".format(rc))


def flip_cross_cols(M, flip_odd=True):
    M_tmp = np.array(M)
    if flip_odd == True:
        a = 1
    else:
        a = 0
    M_tmp[:, a::2] = M_tmp[::-1, a::2]
    if isinstance(M, list):
        return M_tmp.tolist()
    return M_tmp


def flip_cross_rows(M, flip_odd=True):
    """Reverses the order of the elements in odd or even rows.

    Args:
        M (n-by-m matrix): Input matrix
        flip_odd (bool, optional): Whether to reverse the odd or even rows. Defaults to True (odd rows)     .

    Returns:
        M: New matrix with odd or even rows flipped
    """
    M_tmp = np.array(M)
    if flip_odd == True:
        a = 1
    else:
        a = 0
    M_tmp[a::2, :] = M_tmp[a::2, ::-1]
    if isinstance(M, list):
        return M_tmp.tolist()
    return M_tmp


def lists_have_same_elements(a: Iterable, b: Iterable) -> bool:
    """Whether two lists only contains the same elements (not necessarily in the same quantities)

    Args:
        a (Iterable): the first list
        b (Iterable): the second list

    Returns:
        bool: Whether both lists contain the same elements
    """
    return collections.Counter(a) == collections.Counter(b)


def hamming_weight(n: Union[int, str]) -> int:
    """Counts the number of 1s in a binary number. Can input either a binary or int representation

    Args:
        n (str i.e. binary representation or int representation): the number to be processed

    Returns:
        int: the hamming weight, i.e. the number of 1s in the number n
    """
    if isinstance(n, int):
        n = bin(n)
    elif not isinstance(n, str):
        raise TypeError(
            "expected a binary number or an int but got a {}".format(type(n))
        )
    return sum((1 for j in n if j == "1"))


def index_bits(a: str, ones=True) -> list:
    """Takes a binary number and returns a list of indices where the bit is one (or zero)

    Args:
        a (binary number): The binary number whose ones or zeroes will be indexed
        ones (bool): If true, index ones. If false, index zeroes

    Returns:
        list: List of indices where a is one (or zero)
    """
    if isinstance(a, int):
        a = bin(a)
    b = a.split("b")[1]
    if ones:
        return [idx for idx, v in enumerate(b) if int(v)]
    elif not ones:
        return [idx for idx, v in enumerate(b) if not int(v)]


def arg_alternating_index_to_sector(index: int, N: int):
    """This takes in an index and length of the array and returns the index of
    the equivalent sectorized matrix. The argsort equivalent to alternating_indices_to_sectors

    Args:
        index (int): the index to be sectorized
        N (int): the vector length

    Returns:
        int: The sectorized index
    """
    return int(
        (np.ceil(N / 2).astype(int) - 1 + (index + 1) / 2) * (index % 2)
        + (1 - index % 2) * (index / 2)
    )


def arg_alternating_indices_to_sectors(indices: tuple, N: int) -> tuple:
    """The arg equivalent to alternating indices to sectors

    Args:
        indices (tuple): the indices to be moved around
        N (int): the length of the vectors

    Raises:
        ValueError: if the index list doens't have N elements
        TypeError: if N is not the right type

    Returns:
        tuple: tuple of indices corresponding to the transformation
    """
    if isinstance(N, tuple) or isinstance(N, list):
        if len(N) != len(indices):
            raise ValueError(
                "The length of N is not equal to the length of the indices vector"
            )
        return tuple(map(arg_alternating_index_to_sector, indices, N))
    elif isinstance(N, int):
        return tuple(map(arg_alternating_index_to_sector, indices, [N] * len(indices)))
    else:
        raise TypeError(
            "Expected N to be either a tuple or an int, got a {}".format(type(N))
        )


def arg_flip_cross_row(x: int, y: int, dimy: int, flip_odd: bool = True):
    """The arg equivalent of flip cross rows

    Args:
        x (int): the x index of the matrix
        y (int): the y index of the matrix
        dimy (int): the y dimension of the matrix
        flip_odd (bool, optional): Whether to flip the odd or even row indices. Defaults to True.
    """
    if x < 0 or y < 0 or y >= dimy or dimy <= 0:
        raise ValueError(
            "Expected positives indices and dimension, got x:{x},y:{y},dimy:{dimy}".format(
                x=x, y=y, dimy=dimy
            )
        )
    if flip_odd:
        a = 1
        b = 0
    else:
        a = 0
        b = 1
    if x % 2 == a:
        return x, dimy - 1 - y
    elif x % 2 == b:
        return x, y


def grid_to_linear(
    x: int, y: int, dimx: int, dimy: int, horizontal: bool = True
) -> int:
    """Returns the single ravelled index corresponding to a position on a dimx-by-dimy grid given by x and y

    Args:
        x (int): the row index
        y (int): the column index
        dimx (int): the number of rows
        dimy (int): the number of columns
        horizontal (bool, optional): whether the index is row or column major. Defaults to True.

    Returns:
        int: the ravelled index
    """
    if horizontal:
        return x * dimy + y
    else:
        return y * dimx + x


def linear_to_grid(n: int, dimx: int, dimy: int, horizontal: bool = True) -> np.ndarray:
    """Unravels an index into two given two dimensions and the index.

    Args:
        n (int): The initial index
        dimx (int): Number of rows
        dimy (int): Number of columns
        horizontal (bool, optional): Whether to consider n as being. Defaults to True.

    Returns:
        np.ndarray: The two indices corresponding to the original unravelled index
    """

    if horizontal:
        return np.unravel_index((n), (dimx, dimy), order="C")
    else:
        return np.unravel_index((n), (dimx, dimy), order="F")


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalizes a vector by its Frobenius norm

    Args:
        v (np.ndarray): the vector

    Returns:
        np.ndarray: the normalized vector
    """
    return v / np.linalg.norm(v)


def sum_divisible(l: list, i: int):
    """Returns the amount of numbers divisible by an integer i in a list

    Args:
        l (list): the list of integers
        i (int): the interger that is a divisor of the summed numbers

    Returns:
        int: the sum of divisble integers in the list
    """
    return sum([1 if x % i == 0 else 0 for x in l])


def sum_even(l: Iterable) -> int:
    """Returns the number of even numbers in a list

    Args:
        l (Iterable): the list

    Returns:
        int: the number of even numbers
    """
    return sum_divisible(l, 2)


def sum_odd(l: Iterable) -> int:
    """Returns the number of odd numbers in a list

    Args:
        l (Iterable): the list

    Returns:
        int: the number of odd numbers
    """
    return len(l) - sum_even(l)


def ensure_fpath(fpath: os.PathLike):
    """If the file path doesn't exist, create the necessary directories

    Args:
        fpath (os.PathLike): the file path (with the file at the end)
    """
    dirname = os.path.dirname(fpath)
    # turn $HOME and such into actual paths.
    dirname = os.path.expandvars(dirname)
    if dirname != "":
        if not os.path.exists(dirname):
            os.makedirs(dirname)


def normalize_str(s: str, token: str = "") -> str:
    """Replaces all non alphanumeric symbols in a string with token

    Args:
        s (str): the string to be normalized
        token (str, optional): the token which will replace all occurrences of non-alphanumeric symbols. Defaults to "".

    Returns:
        str: the normalized string
    """
    return re.sub(r"\W|\_", token, s)


def cap_first(s: str) -> str:
    """Capitalizes the first letter of a string. Useful for figure captions

    Args:
        s (str): the string whose first letter will be capitalized

    Returns:
        str: The string with the first letter capitalized
    """
    if len(s) == 1:
        return s[0].upper()
    return s[0].upper() + s[1:]


def random_word(lenw=5, Nwords=3) -> str:  # pragma: no cover
    """Generates a random string from either the local dictionary if it is present, or simply from the alphabet

    Args:
        lenw (int, optional): length of words in string. Defaults to 5.
        Nwords (int, optional): number of words in string. Defaults to 3.

    Returns:
        str: the random string
    """
    try:
        dict_paths = ["/usr/share/dict/words", "/usr/dict/words"]
        word_files = [dp for dp in dict_paths if os.path.isfile(dp)]
        word_file = word_files[0]
        words = io.open(word_file, mode="r", encoding="utf8").read().splitlines()
        words = [normalize_str(w) for w in words if len(w) == lenw]
        np.random.seed(seed=None)
        indices = np.random.choice(len(words), Nwords, replace=False)
        word = "".join([cap_first(words[iii]) for iii in indices])
    except:
        word = "".join(
            [
                chr(x)
                for x in np.random.choice(
                    range(97, 97 + 26), lenw * Nwords, replace=True
                )
            ]
        )
    return word


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Returns the quantum fidelity between two objects, each of with being either a wavefunction (a vector) or a density matrix

    Args:
        a (np.ndarray): the first object
        b (np.ndarray): the second object

    Raises:
        ValueError: if there is a mismatch in the dimensions of the objects
        ValueError: if a tensor has more than 2 dimensions

    Returns:
        float: the fidelity between the two objects
    """
    # remove empty dimensions
    squa = np.squeeze(a)
    squb = np.squeeze(b)
    # check for dimensions mismatch
    if len(set((*squa.shape, *squb.shape))) > 1:
        raise ValueError("Dimension mismatch: {} and {}".format(squa.shape, squb.shape))
    # case for two vectors
    if len(squa.shape) == 1 and len(squb.shape) == 1:
        return np.sqrt(
            np.abs(np.dot(np.conj(squa), squb) * np.dot(np.conj(squb), squa))
        )
    else:
        # case for one matrix and one vector, or two matrices
        items = []
        for item in (squa, squb):
            if len(item.shape) == 1:
                items.append(np.outer(item, item))
            elif len(item.shape) == 2:
                items.append(item)
            else:
                raise ValueError(
                    "expected vector or matrix, got {}dimensions".format(item.shape)
                )

        items[0] = sqrtm(items[0])
        print(items)
        rho_sigma_rho = pi_matmul(items[0], items[1], items[0])
        final_mat = sqrtm(rho_sigma_rho)
        return np.trace(final_mat) ** 2


def infidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the infidelity between two wavefunctions or density matrices (or any combination thereof)

    Args:
        a (np.ndarray): the first wavefunction or density matrix
        b (np.ndarray): the second wavefunction or density matrix

    Returns:
        float: the infidelity between a and b
    """
    return 1 - fidelity(a, b)


def save_to_json(
    data,
    dirname: str = None,
    fname: str = None,
    randname: bool = False,
    date: bool = True,
):
    """Saves data to json with more steps for convenience

    Args:
        data (jsonifiable object): data to be saved
        dirname (str, optional): directory to save the object to. Defaults to None.
        fname (str, optional): name of the json file. Defaults to None.
        randname (bool, optional): whether to append a random string to the name. Defaults to False.
        date (bool, optional): whether to append the date to the string. Defaults to True.
    """
    sobj = fauvqe.json.dumps(data, ensure_ascii=False, indent=4)

    if fname is None:
        fname = random_word()
    elif randname:
        fname = fname + "_" + random_word()
    if dirname is not None:
        fpath = os.path.join(dirname, fname)
    else:
        fpath = fname

    if date:
        fpath = fpath + "_" + now_str()

    ensure_fpath(fpath)
    if not fpath.endswith(".json"):
        fpath = fpath + ".json"
    fout = io.open("{fpath}".format(fpath=fpath), "w+", encoding="utf8")
    fout.write(sobj)
    fout.close()
    print("saved {}".format(fpath))


def grid_neighbour_list(
    i: int,
    shape,
    neighbour_order: int,
    periodic: bool,
    diagonal: bool,
    origin: str = "center",
):
    """Creates a list of indices which are the nearest neighbour to some index on a grid. The initial index is assumed to be unraveled

    Args:
        i (int): Initial index
        shape (tuple): shape of the grid
        neighbour_order (int): order of nearest neighbour
        periodic (bool): whether to consider periodic boundary conditions or open
        diagonal (bool): whether to consider diagonal neighbours as well

    Returns:
        list: a list of unraveled (wrt to the grid) indices which correspond to the nearest neighbours
    """
    numrows, numcols = shape
    mat = np.reshape(np.arange(0, numrows * numcols), (numrows, numcols))
    grid = []
    m, n = np.unravel_index(i, (numrows, numcols))

    rowup = 0
    colleft = 0
    rowdown = neighbour_order
    colright = neighbour_order
    if origin == "center":
        rowup = -neighbour_order
        colleft = -neighbour_order
    elif origin == "topleft":
        pass

    for j in range(m + rowup, m + rowdown + 1):
        for k in range(n + colleft, n + colright + 1):
            if periodic:
                j = j % numrows
                k = k % numcols
            if j < numrows and k < numcols and j >= 0 and k >= 0:
                if diagonal:
                    grid.append(mat[j, k])
                elif j == m or k == n:
                    grid.append(mat[j, k])
    return grid


def default_value_handler(shape: tuple, value: Union[str, float]):
    """General function to have some consistency in default value handling names

    Args:
        shape (tuple): shape of the array
        value (Union[str,float]): name of the option

    Returns:
        np.ndarray: array of specifeid default value
    """
    if isinstance(value, float):
        return np.full(shape=shape, fill_value=value)
    if value == "zeros":
        return np.zeros(shape=shape)
    if value == "ones":
        return np.ones(shape=shape)
    if value == "random":
        return np.random.rand(*shape)
    else:
        raise ValueError("expected a valid option, got {}".format(value))


def wrapping_slice(arr: Iterable, indices: list):
    """Get a slice of some array (can be str) given some indices which wraps around if the index list is longer than the array

    Args:
        arr (list): array to be sliced
        indices (list): list of indices which will pick the elements in the array

    Returns:
        list: array slice with given indices
    """
    out_arr = [arr[i % len(arr)] for i in indices]
    if isinstance(arr, str):
        return "".join(out_arr)
    return out_arr


def jw_computational_wf(indices: list, Nqubits: int) -> np.ndarray:
    """Creates a 2**Nqubits wavefunction corresponding to the computational state of Nqubits with the qubits in indices set to one

    Args:
        indices (list): the indices of the qubits which are non-zero
        Nqubits (int): the number of qubits in the wavefunction

    Returns:
        np.ndarray: 2**Nqubits vector with a one in the entry correspknding to the desired computational state
    """
    wf = np.zeros((2**Nqubits))
    jw_index = sum((2 ** (iii) for iii in indices))
    wf[jw_index] = 1
    return wf
