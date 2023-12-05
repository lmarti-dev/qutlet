"""
    Implementation of the fidelity as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict, Optional, Union
from numbers import Integral, Real

import numpy as np

import qutip
from cirq import fidelity

from fauvqe.objectives.objective import Objective
from fauvqe.models.qubitModel import AbstractModel


class Fidelity(Objective):
    """
    Fidelity objective

    This class implements as objective the fidelity of two states given as a pure state vector or as a density matrix

    Parameters
    ----------
    model: AbstractModel, The linked model
    options:    "target_state"    -> np.ndarray    target state to calculate fidelity with

    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <Fidelity target state=self._target_state>

    evaluate(self, wavefunction) : np.float64
        Returns
        ---------
        np.float64:
            if(self.pure):
                target^dagger @ wavefunction
            else:
                qutip.metrics.fidelity(target, wavefunction)
    """

    def __init__(self, model: AbstractModel, target_state: Union[np.ndarray, qutip.Qobj]):
        super().__init__(model)
        self._n = np.size(model.qubits)
        self.set_target_state(target_state)

    def set_target_state(self, target_state: Union[np.ndarray, qutip.Qobj]) -> None:
        self._target_state = target_state

    def evaluate(self, wavefunction: np.ndarray, target_state: np.ndarray = None) -> np.float64:
        if target_state is None:
            target_state = self._target_state
        n_qubits = len(self.model.qubits)
        qid_shape = (2,) * n_qubits
        # cirq fidelity is the square of qutip fidelity
        return fidelity(target_state, wavefunction, qid_shape=qid_shape)

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "target_state": self._target_state,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<Fidelity target_state={}>".format(self._target_state)


class Infidelity(Fidelity):
    def evaluate(self, wavefunction: Union[np.ndarray, qutip.Qobj]) -> np.float64:
        return 1 - super().evaluate(wavefunction=wavefunction)
