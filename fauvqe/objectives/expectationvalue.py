"""
    Implementation of the expectation value of the model hamiltonian as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict
from numbers import Integral

import numpy as np

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel
from fauvqe.objectives.abstractexpectationvalue import AbstractExpectationValue


class ExpectationValue(AbstractExpectationValue):
    """Energy expectation value objective

    This class implements as objective the expectation value of the energies
    of the linked model.

    Parameters
    ----------
    model: AbstractModel    The linked model
    
    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <ExpectationValue Energy fields=self.__energy_fields>
    """

    def __init__(self, model: AbstractModel):
        super().__init__(model)
        self.__energy_fields: List[Literal["X", "Y", "Z"]] = model.energy_fields
        self.__energies: Tuple[np.ndarray, np.ndarray] = model.energy()
        self.__n_qubits: Integral = np.log2(np.size(self.__energies[0]))
    
    def evaluate(self, wavefunction: np.ndarray, options: dict = {}) -> np.float64:
        assert len(self.__energy_fields) == len(self.__energies), "Length of Pauli types and energy masks do not match, {}!={}".format(len(self.__energy_fields), len(self.__energies))
        print(self.__energies)
        i=0
        res=0
        for field in self.__energy_fields:
            if(field == "X"):
                rot_wf = self._rotate_x(wavefunction)
            elif(field == "Y"):
                rot_wf = self._rotate_y(wavefunction)
            elif(field == "Z"):
                rot_wf = wavefunction
            else:
                raise NotImplementedError()
            
            res += np.sum( np.abs(rot_wf) ** 2 * (-self.__energies[i]) ) / self.__n_qubits
            i+=1
        return res
    
    #Old Version
    """
    def evaluate(self, wavefunction: np.ndarray, options: dict = {}) -> np.float64:
        if self.__field == "X":
            wf_x = self._rotate_x(wavefunction)

            return (
                np.sum(
                    np.abs(wavefunction) ** 2 * (-self.__energies[0])
                    + np.abs(wf_x) ** 2 * (-self.__energies[1])
                )
                / self.__n_qubits
            )
        
        # field must be "Z"
        return (
            np.sum(np.abs(wavefunction) ** 2 * (-self.__energies[0] - self.__energies[1]))
            / self.__n_qubits
        )
    """

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<ExpectationValue Energy field={}>".format(self.__energy_fields)

    def __eq__(self, other): 
        '''Temporary solution'''
        if not isinstance(other, self.__class__):
            # don't attempt to compare against unrelated types
            return False

        #Most general: avoid to define Attributes
        temp_bools = []
        for key in self.__dict__.keys():
            print(key)
            if(key == '_ExpectationValue__energies'):
                temp_bools.append((getattr(self, key)[0] == getattr(other, key)[0]).all())
                temp_bools.append((getattr(self, key)[1] == getattr(other, key)[1]).all())
                continue
            temp_bools.append(getattr(self, key) == getattr(other, key))
        return all(temp_bools)