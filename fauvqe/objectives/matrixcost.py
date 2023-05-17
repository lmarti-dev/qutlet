"""
    Implementation of the Frobenius distance between two matrices as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict, Optional
from numbers import Integral, Real

import numpy as np

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel
import cirq

class MatrixCost(Objective):
    """
    Matrix/Vector cost objective

    This class implements as objective the "difference" between an exact Matrix/Vector and 
    a unitary matrix or state vector given by the circuit of an AbstractModel
    Parameters
    ----------
    model: AbstractModel, The linked model
    options:    "MatrixCost" or "VectorCost" -> Determin by dimension of 

    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <MatrixCost matrix=self._matrix>
    """
    def __init__(   self,
                    model: AbstractModel, 
                    matrix: np.ndarray):
        super().__init__(model)
        self._matrix = matrix
        self._N = matrix.shape[0]  
        assert(np.log2(self._N).is_integer()),\
            "MatrixCostError in __init__: matrix dimension not 2^n, received {}".format(self._N)
        self._n=np.log2(self._N)

        if len(matrix.shape) == 2:
            assert(matrix.shape[0] == matrix.shape[1]),\
                "MatrixCostError in __init__: expected square matrix, received {}".format(len(matrix.shape))
            self.__IsVec = False 
        elif len(matrix.shape) == 1:
            self.__IsVec = True
        else:
            assert(False),"MatrixCostError in __init__: expected 1D or 2D tensor, received {}".format(len(matrix.shape))

    def evaluate(self, wavefunction: np.ndarray) -> np.longdouble:
        if self.__IsVec == False:
            #Calculation via Forbenius norm
            #Then the "wavefunction" is also a unitary matrix
            return 1 - abs(np.trace(np.matrix.getH(self._matrix) @ wavefunction)) / self._N
        else:
            #Calculation of the overlap of the given wavefunction with
            #vector self._matrix
            return 1 - abs( np.vdot(self._matrix, wavefunction))

    #Need to overwrite simulate from parent class in order to work
    def simulate(self, param_resolver, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        #return unitary if self.__IsVec == False
        if self.__IsVec == False:
            return cirq.resolve_parameters(self._model.circuit, param_resolver).unitary()
        else:
            return super().simulate(param_resolver, initial_state)

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "matrix": self._matrix,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<MatrixCost matrix={}>".format(self._matrix)