from __future__ import annotations
import abc
from cirq import PauliSum as cirq_PauliSum
from cirq import PauliString as cirq_PauliString
from cirq import X as cirq_X
from cirq import Z as cirq_Z
import importlib
import joblib
import numpy as np
from PyTrilinos import Epetra; PyTrilinos_supported = True
from sys import stdout
from scipy.sparse import csr_matrix as scipy_csr_matrix
from typing import Dict

#try:
#    from PyTrilinos import Epetra
#    PyTrilinos_supported = True
#except:
#    PyTrilinos_supported = False

class Restorable(abc.ABC):
    def __init__(self):
        self.PyTrilinos_Support =  PyTrilinos_supported

    @abc.abstractmethod
    def to_json_dict(self) -> Dict:
        raise NotImplementedError()  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def from_json_dict(cls, params: Dict):
        raise NotImplementedError()  # pragma: no cover
   
    def __eq__(self, other): 
        if not isinstance(other, self.__class__):
            # don't attempt to compare against unrelated types
            return False
        
        #Most general: avoid to define Attributes
        temp_bools = []
        for key in self.__dict__.keys():
            #print(key)
            if isinstance(getattr(self, key), np.ndarray):
                if isinstance(getattr(other, key), np.ndarray):
                    if len(getattr(self, key)) != 0 and len(getattr(other, key)) != 0:
                        #print("key: \t{}\n(getattr(self, key): \n{}\ngetattr(other, key): \n{}\n".format(key, getattr(self, key), getattr(other, key)))
                        temp_bools.append((getattr(self, key) == getattr(other, key)).all())
                    else:
                        temp_bools.append(len(getattr(self, key)) == len(getattr(other, key))) 
                else:
                    return False
            else:
                exceptions = ['simulator', '_get_gradients', '_get_single_cost', 'evaluate', '_optimise_step']
                if key not in exceptions:
                    #print("key: \t{}\ngetattr(self, key): \n{}\ngetattr(other, key): \n{}\n".format(key, getattr(self, key), getattr(other, key)))
                    temp_bools.append(getattr(self, key) == getattr(other, key))
                else:
                    temp_bools.append(getattr(self, key).__class__ == getattr(other, key).__class__)
        #print(temp_bools)
        return all(temp_bools)
    
    def create_range(self, max_index: int, use_progress_bar: bool = False):
        if(use_progress_bar):
            self._tqdm = importlib.import_module("tqdm").tqdm
            return self._tqdm(range(max_index), file=stdout)
        else:
            return range(max_index)

    def epetra_crsmatrix2numpy(self, crsmatrix: Epetra.CrsMatrix):
        N = crsmatrix.NumGlobalCols()
        np_array = np.zeros((N,N))
        map = crsmatrix.Map()
        for i in map.MyGlobalElements():
            Values, Indices = crsmatrix.ExtractGlobalRowCopy(i)
            for j in range(len(Indices)):
                #print("{}\t{}\t{}".format(i, Indices[j], Values[j]))
                np_array[i, int(Indices[j])] = Values[j]
        return np_array

    def numpy2epetra_crsmatrix(self, np_matrix: np.ndarray):
        assert(np.sum(np.iscomplex(np_matrix)) == 0), "Epetra only supports real valued matrices"
        
        # This implementation is probably bad
        # Mostly not clear how to use Epetra.MPIComm 
        Comm  = Epetra.PyComm()
        NumGlobalRows = np_matrix.shape[0]
        Map   = Epetra.Map(NumGlobalRows, 0, Comm)
        non_zero_counts = np.count_nonzero(np_matrix, axis=0)
        maxEntriesperRow = np.amax(non_zero_counts)
        minEntriesperRow = np.amin(non_zero_counts)
        #print("maxEntriesperRow: {}\tminEntriesperRow: {}".format(maxEntriesperRow, minEntriesperRow))
        if maxEntriesperRow - minEntriesperRow < 2:
            crsmatrix     = Epetra.CrsMatrix(Epetra.Copy, Map, maxEntriesperRow, True)
        else:
            crsmatrix     = Epetra.CrsMatrix(Epetra.Copy, Map, 0)
        NumMyRows = Map.NumMyElements()

        for ii in range(NumMyRows):
            # `i' is the global ID of local ID `ii'
            i = Map.GID(ii)

            Indices = np.ndarray.astype(np.flatnonzero(np_matrix[i]), np.int32)
            Values  = np_matrix[i, Indices]
            crsmatrix.InsertGlobalValues(i, Values.astype(np.float64), Indices)
        # transform the matrix into local representation -- no entries
        # can be added after this point. However, existing entries can be
        # modified.
        ierr = crsmatrix.FillComplete()
        # synchronize processors
        Comm.Barrier()

        return crsmatrix

    def cirq_paulisum2scipy_crsmatrix(self, paulisum: cirq_PauliSum, dtype=np.complex128 ):
        #Ensure that qubit ordering is the same 
        _n=len(paulisum.qubits)
        _N = 2**_n
        _qubit_map = {paulisum.qubits[k]: int(k) for k in range(_n)}
        
        x_sparse = scipy_csr_matrix((_N, _N))
        ham_x = cirq_PauliSum()
        __has_x = False
        
        zz_sparse = scipy_csr_matrix((_N, _N))
        ham_z = cirq_PauliSum()
        __has_z = False

        #First seperate PauliSum in X and Z part and raise error if not of ZZ-X-Z form
        for vec, coeff in paulisum._linear_dict.items():
            if type(*set(dict(vec).values())) == type(cirq_X):
                ham_x += coeff*cirq_PauliString(qubit_pauli_map=dict(vec))
                __has_x = True
            elif type(*set(dict(vec).values())) == type(cirq_Z):
                ham_z += coeff*cirq_PauliString(qubit_pauli_map=dict(vec))
                __has_z = True
            else:
                assert False, "Hamiltonian not of ZZ-Z-X form"
        
        if __has_x:
            x_sparse = self._x2scipy_crsmatrix(ham_x, _qubit_map, dtype)
        if __has_z:
            zz_sparse = self._zz2scipy_crsmatrix(ham_z, _qubit_map, dtype)
        
        return zz_sparse + x_sparse



    def _x2scipy_crsmatrix(self, paulisum: cirq_PauliSum, _qubit_map, dtype=np.complex128 ):
        _n = len(paulisum.qubits)
        _N = 2**_n

        # Need to get _n dim coefficent array according to the qubit map 
        # tp be copied _N times
        _coeffs = np.zeros(_n, dtype=dtype) 
        for vec, coeff in paulisum._linear_dict.items():
            tmp= list(dict(vec).keys())[0]
            if np.iscomplexobj(dtype(0)):
                _coeffs[_qubit_map[tmp]] = dtype(coeff) 
            else:
                _coeffs[_qubit_map[tmp]] = dtype(coeff.real) 
        #print(_coeffs)

        # Only need to calc column position 
        # express i binary and the need to exchage on the jth position 0 <-> 1
        # Make this parallel via joblib
        #col = np.empty(_N*_n, dtype=int) 
        #for i in range(_N):
        #    for j in range(_n):
                # maybe better to use np.bitwise (i, 2**j)
                #_binary = list(np.binary_repr(i, width=_n))
                #_binary[j] = str((int(_binary[j])+1)%2)
                #col[j+i*_n]= int(''.join(_binary),2)
                
                # This is approx 30 % faster:
        #        col[j+i*_n] = np.bitwise_xor(i, 2**(_n-j-1))
        
        # This is incomparably faster (10^3ish):
        col = np.bitwise_xor(np.repeat(np.arange(_N), _n), np.tile(2**(_n-np.arange(_n)-1), _N), dtype=int) 

        #np.repeat([1,2],2) ->[1,1,2,2]
        #np.tile([1,2],2)   ->[1,2,1,2]
        return scipy_csr_matrix((np.tile(_coeffs, _N), (np.repeat(np.arange(_N), _n).astype(int), col)), shape=(_N, _N))

    def _zz2binary_fct(self, paulisum: cirq.PauliSum, _qubit_map = dict(), dtype=np.complex128 ):
        _n = len(paulisum.qubits)
        _N = 2**_n
        _qubit_map = {paulisum.qubits[k]: int(k) for k in range(_n)}

        _coeffs = []
        _indices_coeffs = []
        for vec, coeff in paulisum._linear_dict.items():
            tmp= list(dict(vec).keys())
            _indices_coeffs.append([_qubit_map[tmp[i]] for i in range(len(vec))])

            if np.iscomplexobj(dtype(0)):
                _coeffs.append(dtype(coeff))
            else:
                _coeffs.append(dtype(coeff.real))

        def _zz_binary_fct(int_in, _n):
            #assume 0, +1 input
            #potentially speed that up via np routine
            binary_vec = np.array([int(i) for i in bin(int_in)[2:].zfill(_n)])
            _tmp = 0
            for i in range(len(_coeffs)):
                _tmp += _coeffs[i] * np.prod(-2*binary_vec[_indices_coeffs[i]]+1)
            return _tmp
        return _zz_binary_fct

    def _zz2scipy_crsmatrix(self, paulisum: cirq.PauliSum, _qubit_map, dtype=np.complex128 ):
        _n = len(paulisum.qubits)
        _N = 2**_n

        #Get binary function
        _binary_fct = self._zz2binary_fct( paulisum, _qubit_map, dtype)
        
        if _n < 11:
            _binary_fct_vec = np.vectorize(_binary_fct)
            _data = _binary_fct_vec(np.arange(_N), _n)
        else:
            _data= np.array(joblib.Parallel(n_jobs=8, backend="loky")(
                joblib.delayed(_binary_fct)(j, _n)
                for j in range(_N)), dtype=dtype)
        
        #Remove zeros
        non_zeros = np.nonzero(_data)
        return scipy_csr_matrix((_data[non_zeros], (*non_zeros, *non_zeros)), shape=(_N, _N))