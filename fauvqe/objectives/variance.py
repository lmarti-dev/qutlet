"""
    Implementation of the Variance as objective function for an AbstractModel object.

    It requires a wavevector with respect to which the variance is calculated

"""
import cirq
from numbers import Integral
import numpy as np
from typing import Literal, Tuple, Dict, Mapping, Optional, List, Union

from fauvqe.objectives.abstractexpectationvalue import AbstractExpectationValue
from fauvqe.models.abstractmodel import AbstractModel
from fauvqe.objectives.objective import Objective


class Variance(AbstractExpectationValue):
    """
    Varianceobjective

    This class implements as objective the Variance of given observables w.r.t. a given state.

    Parameters
    ----------
    model:          AbstractModel                           The linked model
    observables:    cirq.PauliSum or List[cirq.PauliSum]    Observables of which the variance is calculated
    wavefunction:   np.ndarray                              Optionally store state respect to which the variance is calculated
    
    Methods
    ----------
    evaluate(): 
        observables     optionally provide observables different to the stored one(s)
        wavefunction    optionally provide a wavefunction different to the stored one
        q_map:          optionally provide an alternative qubit order

        Returns np.float64, List[np.float64], the variance values

    __repr__() : str
        Returns
  
        str:
            <Variance observable=self.observable>
    """

    def __init__(   self, 
                    model: AbstractModel, 
                    observables: Optional[Union[cirq.PauliSum, List[cirq.PauliSum]]]=None, 
                    wavefunction: np.ndarray=None,):
        #Optionally store wavevector w.r.t. which variance is calculated

        self._wavefunction=wavefunction
        super().__init__(model, observables)
    
    #ToDo unfy order here with AbStractExpectationValue
    def evaluate(   self, 
                    observables: Optional[Union[cirq.PauliSum, List[cirq.PauliSum]]]=None,
                    wavefunction: np.ndarray=None, 
                    _qubit_order: Mapping[cirq.ops.pauli_string.TKey, int]=None,
                    atol = 1e-7)-> Union[np.float64, List[np.float64]]:
        #Returns variance of observables with respect to given wavefunction
        #E.g. variance of X with respect to X or Z eigen state
        #or variance of ground state |\psi_0> w.r.t. a subsystem Hamiltonian parition
        #Require cirq simulator, as currently qsim gives segmentation fault
        assert(isinstance(self._model.simulator, cirq.Simulator)), "Variance evalute currently requires cirq simulator due to segmentation fault of qsim"
 
        if(observables is None):
            observables = self._observable
        
        if(wavefunction is None):
            if(self._wavefunction is None):
                # This sets the initial wavefunction to |00 ... 00 >
                if isinstance(self._model.n,int):
                    _N=2**self._model.n
                else:
                    _N=2**np.multiply(*self._model.n)
                wavefunction = np.zeros((_N) ,dtype= np.complex64)
                wavefunction[0]=1
            else:
                # Use the stored wavefunction if no other wavefunction was given
                wavefunction = self._wavefunction.view()

        # Defined in AbstractExpectationValue already:
        #if(_qubit_order is None):
        #    _qubit_order = {self._model.qubits[k][l]: int(k*self._model.n[1] + l) for l in range(self._model.n[1]) for k in range(self._model.n[0])}

        #Implement <O²> - <O>²
        if isinstance(observables, list):
            print([observable for observable in observables])
            return [(super().evaluate(wavefunction= wavefunction,
                                q_map = _qubit_order,
                                atol = atol,
                                observable = observable**2)
                    -super().evaluate(wavefunction= wavefunction,
                                q_map = _qubit_order,
                                atol = atol,
                                observable = observable)**2) 
                    for observable in observables]
            square_observables = [observables[i]**2 for i in range(len(observables))]
        else:
            return (super().evaluate(wavefunction= wavefunction,
                                q_map = _qubit_order,
                                atol = atol,
                                observable = observables**2)
                -super().evaluate(wavefunction= wavefunction,
                                q_map = _qubit_order,
                                atol = atol,
                                observable = observables)**2)

        #evaluate(   self, 
        #            wavefunction: np.ndarray, 
        #            q_map: Mapping[cirq.ops.pauli_string.TKey, int]=None, 
        #            atol: float = 1e-7,
        #            observable = None)
        return (super().evaluate(wavefunction= wavefunction,
                                q_map = _qubit_order,
                                atol = atol,
                                observable = observables**2)
                -super().evaluate(wavefunction= wavefunction,
                                q_map = _qubit_order,
                                atol = atol,
                                observable = observables)**2)
        #This is numerically un reliable:
        #return np.real(np.array(self._model.simulator.simulate_expectation_values(
        #            program = cirq.Circuit(), # to keep previous behaviour use empty circuit
        #            qubit_order = _qubit_order,
        #            observables=square_observables, # This allows for a list of observables
        #            initial_state=wavefunction,
        #            ))
        #            - np.power(
        #                np.array(self._model.simulator.simulate_expectation_values(
        #                    program = cirq.Circuit(), # to keep previous behaviour use empty circuit
        #                    qubit_order = _qubit_order,
        #                    observables=observables, # This allows for a list of observables
        #                    initial_state=wavefunction,
        #                    )),
        #                2))

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "observables": self._observable,
                "wavefunction": self._wavefunction
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<Variance observable={}>".format(self._observable)