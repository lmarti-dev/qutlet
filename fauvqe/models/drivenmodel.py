"""
    This class implements time-dependent models based on time-independent AbstractModels

"""
from __future__ import annotations

import numpy as np
import sympy

from cirq import Circuit, PauliSum
from numbers import Real, Number
from importlib import import_module
from itertools import chain
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
    T:          Periodicity of the drives/expansion time/frequency for Magnus series
    t0:         Initial time, default 0
    t:          Optional[Number]
                Fix the time of the DrivenModel, e.g. for variational optimisation
    j_max:      Trunction order for Fourier Series in Magnus expension

    Methods
    ----------
    TODO Add descriptions here

    __repr__() : str
        Returns
        ---------
        str:
            <DrivenModel ....>

     Typ hinting Functions:
     https://stackoverflow.com/questions/37835179/how-can-i-specify-the-function-type-in-my-type-hints
    """
    basics  = import_module("fauvqe.circuits.basics")
    hea  = import_module("fauvqe.circuits.hea")
    qaoa = import_module("fauvqe.circuits.qaoa")

    def __init__(   self,
                    models: Union[  List[AbstractModel], AbstractModel],
                    drives: Union[  List[Callable[[float], float]], 
                                    Callable[[float], float]],
                    T: Real = 2*sympy.pi,
                    t0: Real = 0,
                    t: Real = None,
                    j_max: int = 10):
        #If an AbstractModel and a function are given -> convert those to lists
        if not isinstance(models, list): models = [models]
        if not isinstance(drives, list): drives = [drives]

        assert len(models) == len(drives)\
            , "Error in DrivenModel initialisation: len(models) needs to be len(drives), {} != {}".format(
                len(models), len(drives)
            )
        self.models = models
        self.drives = drives
        self.T = T
        self.t0 = t0
        self.j_max = j_max
        self._t = t

        #Alternative set hamiltonian for t=0
        self._hamiltonian = PauliSum()
        if t is not None: self._set_hamiltonian(t)

        #For the moment require every model to act on same qubits
        self._init_qubits()
        self.n=self.models[0].n

        self.circuit = Circuit()
        self.circuit_param: List[sympy.Symbol] = []
        self.circuit_param_values: Optional[np.ndarray] = None

        self._set_Vjs()
    
    def _init_qubits(self):
        """
            This function checks whether all models act on the same qubits.
            if so, it sets self.qubits = self.models[0].qubits
            
            Parameters
            ----------
            self.models:     Union[List[AbstractModel], AbstractModel]
                        The driven models

            Sets
            -------
            self.qubits = self.models[0].qubits
        """
        _qubits = self.models[0].qubits
        for i_model in range(1,len(self.models)):
            assert _qubits == self.models[i_model].qubits , "Error in DrivenModel initialisation: All models need to act on same qubits"
        self.qubits = _qubits 

    def energy(self, t: Real):
        """
            This function leverages the energy() functions of self.models
            in order to define an energy filter function for Driven Model
            
            Parameters
            ----------
            self.models:     Union[List[AbstractModel], AbstractModel]
                                The driven models
            self.drives:   Union[  List[Callable[float, float]], 
                                            Callable[float, float]]
                            The drive functions
            t:          Time at which to provide energy_filter of the DrivenModel

            Returns
            -------
            energy_filter: List[np.ndarray]
        """
        energy_filter = []

        #Need to treat first instance differently to creat list
        _tmp = self.models[0].energy()
        for i_filter in range(len(_tmp)):
            energy_filter.append(self.drives[0](t)*_tmp[i_filter])

        for i_model in range(1,len(self.models)):
            _tmp = self.models[i_model].energy()
            for i_filter in range(len(_tmp)):
                energy_filter[i_filter] += self.drives[i_model](t)*_tmp[i_filter]

        return energy_filter

    def hamiltonian(self, t: Real = 0) -> PauliSum:
        """
            This function returns a cirq.PauliSum for a specific time t
            in order to define an energy filter function for Driven Model
            
            Parameters
            ----------
            self.models:     Union[List[AbstractModel], AbstractModel]
                            The driven models

            t:              Time for witch to return the hamiltonian

            Returns
            -------
            hamiltonian:    cirq.PauliSum

            Note that unfortunately cannot define a cirq.PauliSum with function dependencies
            https://quantumai.google/reference/python/cirq/PauliSum
        """
        hamiltonian = PauliSum()
        for i in range(len(self.models)):
            hamiltonian += self.drives[i](t)*self.models[i].hamiltonian
        return hamiltonian

    def K(  self,
            t: Real) -> PauliSum:
        """
            This implements the Kick-operator:
                K(t) = 

            Parameters
            ----------
        """
        K_t =  PauliSum()
        for i_drive in range(len(self.drives)):
            _tmp = sum(self.Vjs[i_drive][i_j]*sympy.exp(sympy.I*2*sympy.pi*(i_j-self.j_max)*t/self.T) for i_j in range(self.j_max)).expand(complex=True)
            _tmp += sum(self.Vjs[i_drive][self.j_max+i_j-1]*sympy.exp(sympy.I*2*sympy.pi*i_j*t/self.T) for i_j in range(1,self.j_max+1)).expand(complex=True)
            #print(sympy.N(_tmp))
            K_t += sympy.N(_tmp, 16)*self.models[i_drive]._hamiltonian

            #i_j = 0
            #for j in chain(range(-self.j_max,0), range(1,self.j_max)):
                #sum(c(n, f, T, t, t0)*sympy.exp(sympy.I*2*sympy.pi*n*t/T) for n in range(-N, N+1)).expand(complex=True).simplify()
            #    K_t += (self.T/(2*sympy.pi*j*sympy.I))*sympy.exp(2*sympy.pi*j*sympy.I/self.T)*self.Vjs[i_drive][i_j]#*self.models[i_drive]._hamiltonian
            #    i_j += 1
        return K_t

    def set_circuit(self, qalgorithm, options: dict = {}):
        """
            Possibly adapt from SpinModel

            To be implemented:
                -   Trotter -> Use potentially set trotter circuit from Model classes
                -   Matchgate (for 1D TFIM)
                -   Kick operator VFF
                -   Floquet normalform
        """
        raise NotImplementedError()

    def _set_hamiltonian(self, t: Real):
        self.hamiltonian = self.hamiltonian(t)

    def _get_Vj(    self,
                    j: int, 
                    drive: Callable[[Real], Real]):
        """
            TODO docstring
        """
        s = sympy.Symbol('s', real=True)
        return sympy.integrate( drive(s)*sympy.exp((sympy.I*2*sympy.pi * j * s)/self.T),
                            (s, self.t0, self.t0 + self.T))/self.T

    def _set_Vjs(self):
        """
            TODO docstring

            Parameters
            ----------
            j_max:      Truncation order for Fourier series
        """
        self.Vjs = []
        for i_drive in range(len(self.drives)):
            self.Vjs.append([self._get_Vj(j,self.drives[i_drive]) for j in chain(range(-self.j_max,0), range(1, self.j_max+1))])

    def set_Ut( self, 
                t: Real,
                m: int = 1000):
        raise NotImplementedError()
    
    def copy(self) -> DrivenModel:
        raise NotImplementedError()

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "models": self.models,
                "drives": self.drives,
                "t": self._t,
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

    def __repr__(self) -> str:
        """
        
            Note that for this to work properly need to set __name__ of drive
            drive0 = lambda t : sympy.sin((2*sympy.pi/T)*t)
            drive0.__name__ = 'f(t) = sin((2pi/T)*t)'
        """
        _str = "< DrivenModel\n"
        for i in range(len(self.models)):
            _str += "Model " + str(i) + " " + repr(self.models[i]) + "\n"
            _str += "Drive " + str(i) + " " + self.drives[i].__name__ + "\n"
        return _str[:-1] + " >"
