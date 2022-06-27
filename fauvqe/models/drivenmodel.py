"""
    This class implements time-dependent models based on time-independent AbstractModels

"""
from numbers import Real
import numpy as np
from sympy  import Symbol
from typing import Callable, List, Optional, Union

from fauvqe.models.abstractmodel import AbstractModel

class DrivenModel(AbstractModel):
    """
    This class implements time-dependent models based on time-independent AbstractModels.
    It generates an AbstractModel of the form:
        \sum_i drive_function_i * model_i

    For example, to obtain a driven Transverse Field Ising Model:
        -Generate a ZZ-Ising object and a X-Ising object
        -Set drive_0 time-independent to lambda t: return 1 and drive_1 e.g. to lambda t: return sin(t)
        -driven_Ising_model = DrivenModel([model_0, model_1], [drive_0, drive_1])

    Parameters
    ----------
    models:     Union[List[AbstractModel], AbstractModel]
                The driven models
    drives:     Union[  List[Callable[float, float]], 
                        Callable[float, float]]
                The drive functions
    t:          Optional[Number]
                Fix the time of the DrivenModel, e.g. for variational optimisation


    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <cVaR field=self.field alpha=self.alpha>

     Typ hinting Functions:
     https://stackoverflow.com/questions/37835179/how-can-i-specify-the-function-type-in-my-type-hints
    """

    def __init__(   self,
                    models: Union[  List[AbstractModel], AbstractModel],
                    drives: Union[  List[Callable[float, float]], 
                                    Callable[float, float]],
                    t: Real = None):
        self._models = models
        self._drives = drives
        self._t = t

        self.circuit = cirq.Circuit()
        self.circuit_param: List[Symbol] = []
        self.circuit_param_values: Optional[np.ndarray] = None

        self._set_hamiltonian()

    


    def energy(self, t):
        pass

    def _set_hamiltonian(self):
        self.hamiltonian = 

    """
    def copy(self) -> DrivenModel:


    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "qubittype": self.qubittype,
                "n": self.n,
                "j_v": self.j_v[:,:,0],
                "j_h": self.j_h[:,:,0],
                "h": self.h[:,:,0],
            },
            "params": {
                "circuit": self.circuit,
                "circuit_param": self.circuit_param,
                "circuit_param_values": self.circuit_param_values,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        inst = cls(**dct["constructor_params"])

        inst.circuit = dct["params"]["circuit"]
        inst.circuit_param = dct["params"]["circuit_param"]
        inst.circuit_param_values = dct["params"]["circuit_param_values"]
        
        return inst
    """