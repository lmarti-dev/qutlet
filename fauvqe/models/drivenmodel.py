"""
    This class implements time-dependent models based on time-independent AbstractModels

"""
from cirq import Circuit, PauliSum
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
        assert len(models) == len(drives)\
            , "Error in DrivenModel initialisation: len(models) needs to be len(drives), {} != {}".format(
                len(models), len(drives)
            )
        self._models = models
        self._drives = drives
        self._t = t

        #For the moment require every model to act on same qubits
        self._init_qubits()

        self.circuit = Circuit()
        self.circuit_param: List[Symbol] = []
        self.circuit_param_values: Optional[np.ndarray] = None

        self._set_hamiltonian()

    


    def energy(self, t: Real):
        """
            This function leverages the energy() functions of self._models
            in order to define an energy filter function for Driven Model
            
            Parameters
            ----------
            self._models:     Union[List[AbstractModel], AbstractModel]
                                The driven models
            self._drives:   Union[  List[Callable[float, float]], 
                                            Callable[float, float]]
                            The drive functions
            t:          Time at which to provide energy_filter of the DrivenModel

            Returns
            -------
            energy_filter: List[np.ndarray]
        """
        energy_filter = []

        #Need to treat first instance differently to creat list
        _tmp = self._models[0].energy()
        for i_filter in range(len(_tmp)):
            energy_filter.append(self._drives[0](t)*_tmp[i_filter])

        for i_model in range(1,len(self._models)):
            _tmp = self._models[i_model].energy()
            for i_filter in range(len(_tmp)):
                energy_filter[i_filter] += self._drives[i_model](t)*_tmp[i_filter]

        return energy_filter

    def _init_qubits(self):
        """
            This function checks whether all models act on the same qubits.
            if so, it sets self.qubits = self._models[0].qubits
            
            Parameters
            ----------
            self._models:     Union[List[AbstractModel], AbstractModel]
                        The driven models

            Sets
            -------
            self.qubits = self._models[0].qubits
        """

    def get_hamiltonian(self, t: Real) -> PauliSum:
        """
            This function returns a cirq.PauliSum for a specific time t
            in order to define an energy filter function for Driven Model
            
            Parameters
            ----------
            self._models:     Union[List[AbstractModel], AbstractModel]
                            The driven models

            t:              Time for witch to return the hamiltonian

            Returns
            -------
            hamiltonian:    cirq.PauliSum

            Note that unfortunately cannot define a cirq.PauliSum with function dependencies
            https://quantumai.google/reference/python/cirq/PauliSum
        """
        hamiltonian = PauliSum()
        for i in range(len(self._models)):
            hamiltonian += self._drives[i](t)*self._models[i].hamiltonian
        return hamiltonian

    def set_circuit(self, qalgorithm, options: dict = {}):
        """
            Possibly adapt from SpinModel

            To be implemented:
                -   Trotter
                -   Matchgate (for 1D TFIM)
                -   Kick operator VFF
                -   Floquet normalform
        """
        raise NotImplementedError()

    def _set_hamiltonian(self, t: Real):
        self.hamiltonian = self.get_hamiltonian(t)

    
    def copy(self) -> DrivenModel:
        raise NotImplementedError()

    def to_json_dict(self) -> Dict:
        raise NotImplementedError()
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
        raise NotImplementedError()
        inst = cls(**dct["constructor_params"])

        inst.circuit = dct["params"]["circuit"]
        inst.circuit_param = dct["params"]["circuit_param"]
        inst.circuit_param_values = dct["params"]["circuit_param_values"]
        
        return inst
