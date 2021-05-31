"""Implementation of the conditional value at risk as objective function for a model, e.g. Ising of AbstractModel type
"""

from typing import Literal, Dict
from numbers import Real, Integral

import numpy as np

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel


class CVaR(Objective):
    """Class implementing the conditional value at risk (cVaR) as an objective.

    Parameters
    ----------
    model: AbstractModel
        The linked model
    alpha: Real
        The :math:`\\alpha` value
    field: {"X", "Z"}, default "Z"
        The field to be evaluated


    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <cVaR field=self.field alpha=self.alpha>
    """

    def __init__(self, model: AbstractModel, alpha: Real, field: Literal["X", "Z"] = "Z"):
        super().__init__(model)

        assert 0.0 <= alpha <= 1.0, "cVaR alpha must be in (0, 1). Received: {:f}".format(alpha)
        assert field in [
            "Z",
            "X",
        ], "Bad input 'field'. Allowed values are ['X', 'Z' (default)], received {}".format(field)

        # Calculate energies of model
        energies_j, energies_h = model.energy()

        self.__alpha: Real = alpha
        self.__field: Literal["X", "Z"] = field
        self.__n_qubits: Integral = np.log2(np.size(energies_h))

        if field == "X":
            self.evaluate = self._evaluate_x

            self.__energies_j = energies_j
            self.__energies_h = energies_h

        if field == "Z":
            self.evaluate = self._evaluate_z

            energies = -energies_j + energies_h
            self.__mask: np.ndarray = np.argsort(energies)
            self.__energies = energies[self.__mask]

    def evaluate(self, wavefunction: np.ndarray) -> Real:
        """Calculate the conditional value at risk (cVaR) for a given wavefunction

        Evaluates the wavefunction with the given :math:`\\alpha` and field.

        The respective field choices are implemented as `CVaR._evaluate_x` and `CVaR._evaluate_z`.

        Parameters
        ----------
        wavefunction: numpy.ndarray
            The input wavefunction in z basis

        Returns
        -------
        Real: The conditional value at risk (cVaR).
        """
        return self._evaluate_z(wavefunction)  # pragma: no cover

    def _evaluate_x(self, wavefunction_z: np.ndarray) -> Real:
        """Calculate the conditional value at risk (cVaR) for a given wavefunction in x basis

        Equivalent to `CVaR.evaluate` when field="X".

        Parameters
        ----------
        wavefunction_z: numpy.ndarray
            THe input wavefunction in z basis

        Returns
        -------
        Real: The conditional value at risk (cVaR) in x basis.
        """
        wavefunction_x = self._rotate_x(wavefunction_z)
        energies = np.abs(wavefunction_z) ** 2 * (-self.__energies_j) + np.abs(
            wavefunction_x
        ) ** 2 * (-self.__energies_h)
        mask = np.argsort(energies)
        _energies = energies[mask]
        _probabilities = (np.abs(wavefunction_x) ** 2)[mask]

        return CVaR._calc_cvar(_probabilities, _energies, self.__alpha) / self.__n_qubits

    def _evaluate_z(self, wavefunction: np.ndarray) -> Real:
        """Calculate the conditional value at risk (cVaR) for a given wavefunction in z basis

        Equivalent to `CVaR.evaluate` when field="Z".

        Parameters
        ----------
        wavefunction: numpy.ndarray
            THe input wavefunction in z basis

        Returns
        -------
        Real: The conditional value at risk (cVaR) in z basis.
        """
        return (
            CVaR._calc_cvar((np.abs(wavefunction) ** 2)[self.__mask], self.__energies, self.__alpha)
            / self.__n_qubits
        )

    @staticmethod
    def _calc_cvar(probabilities: np.ndarray, energies: np.ndarray, alpha: Real) -> Real:
        """Calculate the conditional value at risk (cVaR) for a given distribution.

        Parameters
        ----------
        probabilities: numpy.ndarray
        energies: numpy.ndarray
        alpha: Real

        Returns
        -------
        Real
            The conditional value at risk (cVaR) for the given distribution
        """
        # Expect energies and probabilities to be ordered
        mask = np.cumsum(probabilities) <= alpha

        if not np.any(mask):
            return energies[0]

        energies_ = energies[mask]
        probabilities_ = probabilities[mask]

        return np.sum(energies_ * probabilities_) / np.sum(probabilities_)

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "field": self.__field,
                "alpha": self.__alpha,
                "model": self._model,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<cVaR field={} alpha={}>".format(self.__field, self.__alpha)
