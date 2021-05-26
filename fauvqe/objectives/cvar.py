"""Implementation of the conditional value at risk as objective function for an initialiser
"""

from typing import Literal
from numbers import Real, Integral

import numpy as np

from fauvqe.objectives.objective import Objective
from fauvqe.initialisers.initialiser import Initialiser


class CVaR(Objective):
    """Class implementing the conditional value at risk (cVaR) as an objective.

    Parameters
    ----------
    alpha: Real
        The :math:`\\alpha` value
    field: {"X", "Z"} default "Z"
        The field to be evaluated


    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <cVaR field=self.field alpha=self.alpha>
    """

    def __init__(self, initialiser: Initialiser, alpha: Real, field: Literal["X", "Z"] = "Z"):
        super().__init__(initialiser)

        assert 0.0 <= alpha <= 1.0, "cVaR alpha must be in (0, 1). Received: {:f}".format(alpha)
        assert field in [
            "Z",
            "X",
        ], "Bad input 'field'. Allowed values are ['X', 'Z' (default)], received {}".format(field)

        # Calculate energies of initialiser
        energies_j, energies_h = initialiser.energy()

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

    def _evaluate_x(self, wavefunction_z: np.ndarray) -> Real:
        wavefunction_x = self._rotate_x(wavefunction_z)
        energies = np.abs(wavefunction_z) ** 2 * (-self.__energies_j) + np.abs(
            wavefunction_x
        ) ** 2 * (-self.__energies_h)
        mask = np.argsort(energies)
        _energies = energies[mask]
        _probabilities = (np.abs(wavefunction_x) ** 2)[mask]

        return CVaR._calc_cvar(_probabilities, _energies, self.__alpha)

    def _evaluate_z(self, wavefunction: np.ndarray) -> Real:
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
        return (
            CVaR._calc_cvar((np.abs(wavefunction) ** 2)[self.__mask], self.__energies, self.__alpha)
            / self.__n_qubits
        )

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
