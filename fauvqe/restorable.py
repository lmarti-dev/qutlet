from __future__ import annotations
import abc
import importlib
import numpy as np
from sys import stdout
from typing import Dict

try:
    from PyTrilinos import Epetra
    PyTrilinos_supported = True
except:
    PyTrilinos_supported = False

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

    def Epetra_CrsMatrix2Numpy(self, crsmatrix: Epetra.CrsMatrix):
        N = crsmatrix.NumGlobalCols()
        np_array = np.zeros((N,N))
        map = crsmatrix.Map()
        for i in map.MyGlobalElements():
            Values, Indices = crsmatrix.ExtractGlobalRowCopy(i)
            for j in range(len(Indices)):
                #print("{}\t{}\t{}".format(i, Indices[j], Values[j]))
                np_array[i, int(Indices[j])] = Values[j]
        return np_array

    def Numpy2Epetra_CrsMatrix(self, np_matrix: np.ndarray):
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