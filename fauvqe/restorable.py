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

    def Epetra_CrsMatrix2Numpy(crsmatrix: Epetra.CrsMatrix):
        N = crsmatrix.NumGlobalCols()
        np_array = np.zeros((N,N))
        map = crsmatrix.Map()
        for i in map.MyGlobalElements():
            Values, Indices = crsmatrix.ExtractGlobalRowCopy(i)
            for j in range(len(Indices)):
                #print("{}\t{}\t{}".format(i, Indices[j], Values[j]))
                np_array[i, int(Indices[j])] = Values[j]
        return np_array