"""Implementation of the conditional value at risk as objective function for a model, e.g. Ising of AbstractModel type
"""

from typing import Literal
from numbers import Real, Integral

import numpy as np

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel


class CVaR(Objective):
    """Class implementing the conditional value at risk (cVaR) as an objective.

    Parameters
    ----------
    alpha: Real
        The :math:`\\alpha` value
    field: {"Z"} default "Z"
        The field to be evaluated


    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <cVaR field=self.field alpha=self.alpha>
    """

    def __init__(self, model: AbstractModel, alpha: Real, field: Literal["Z"] = "Z"):
        super().__init__(model)

        assert 0.0 <= alpha <= 1.0, "cVaR alpha must be in (0, 1). Received: {:f}".format(alpha)
        assert field in [
            "Z",
        ], "Bad input 'field'. Allowed values are ['Z' (default)], received {}".format(field)

        self.__alpha: Real = alpha
        self.__field: Literal["Z"] = field

        # Calculate energies of model
        energies_j, energies_h = model.energy()

        # Generate sorting masks
        energies = -energies_j + energies_h
        self.__mask: np.ndarray = np.argsort(energies)
        self.__mask_j: np.ndarray = np.argsort(energies_j)
        self.__mask_h: np.ndarray = np.argsort(energies_h)

        # Sort and store energies
        self.__energies = energies[self.__mask]
        self.__energies_j: np.ndarray = energies_j[self.__mask_j]
        self.__energies_h: np.ndarray = energies_h[self.__mask_h]

        self.__n_qubits: Integral = np.log2(np.size(energies_h))

    def evaluate(self, wavefunction: np.ndarray) -> Real:
        """Calculate the conditional value at risk for a given wavefunction.

        Parameters
        ----------
        wavefunction:
            The wavefunction to calculate the conditional value at risk.

        Returns
        ----------
        Real:
            The conditional value at risk (cVaR)
        """
        return CVaR._calc_cvar(
            (np.abs(wavefunction) ** 2)[self.__mask], self.__energies, self.__alpha
        )

        # Wavefunction is expected to be normalized
        # Reorder wavefunction with stored mask
        # probabilities = np.abs(wavefunction) ** 2
        # probabilities_j = probabilities[self.__mask_j]
        #
        # probabilities_h = probabilities[self.__mask_h]
        #
        # if self.__field == "X":
        #     wavefunction_x = self._rotate_x(wavefunction)
        #     probabilities_h = np.abs(wavefunction_x) ** 2
        #     probabilities_h = probabilities_h[self.__mask_h]
        #
        # Calculate cVaR independently for interaction and field
        # cvar_j = CVaR._calc_cvar(probabilities_j, self.__energies_j, self.__alpha)
        # cvar_h = CVaR._calc_cvar(probabilities_h, self.__energies_h, self.__alpha)
        #
        # if self.__field == "X":
        #     return (-cvar_j - cvar_h) / self.__n_qubits

        # field must be "Z"
        # return (-cvar_j + cvar_h) / self.__n_qubits

    @staticmethod
    def _calc_cvar(probabilities: np.ndarray, energies: np.ndarray, alpha: Real) -> Real:
        # Expect energies and probabilities to be ordered
        mask = np.cumsum(probabilities) <= alpha

        if not np.any(mask):
            return energies[0]

        energies_ = energies[mask]
        probabilities_ = probabilities[mask]

        return np.sum(energies_ * probabilities_) / np.sum(probabilities_)

    def __repr__(self) -> str:
        return "<cVaR field={} alpha={}>".format(self.__field, self.__alpha)
