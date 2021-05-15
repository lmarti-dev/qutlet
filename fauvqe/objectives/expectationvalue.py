"""
Expval module docstring
"""

from typing import Literal, Tuple
import numpy as np
from numbers import Integral

from fauvqe.objectives.objective import Objective
from fauvqe.initialisers.initialiser import Initialiser


class ExpectationValue(Objective):
    """Expectation value objective

    This class implements as objective the expectation value of the energies
    of the linked initialiser.

    Mathmatical description

    :math:`\\frac{1}{N} \\sum_N |\\Psi_\\text{N}|^2 \\cdot E_\\text{field}`

    Parameters
    ----------
    field: {"X", "Z"} default "Z"
        The field to be evaluated

    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <ExpectationValue field=self.field>
    """

    def __init__(self, initialiser: Initialiser, field: Literal["Z", "X"] = "Z"):
        super().__init__(initialiser)
        assert field in [
            "Z",
            "X",
        ], "Bad argument 'field'. Allowed values are ['X', 'Z' (default)], revieced {}".format(
            field
        )

        self.__field: Literal["Z", "X"] = field
        self.__energies: Tuple[np.ndarray, np.ndarray] = initialiser.energy()
        self.__n_qubits: Integral = np.log2(np.size(self.__energies[0]))

    def evaluate(self, wavefunction: np.ndarray) -> np.float64:
        if self.__field == "X":
            wf_x = self._rotate_x(wavefunction)

            return (
                np.sum(
                    np.abs(wavefunction) ** 2 * (-self.__energies[0])
                    + np.abs(wf_x) ** 2 * (-self.__energies[1])
                )
                / self.__n_qubits
            )
        if self.__field == "Z":
            return (
                np.sum(np.abs(wavefunction) ** 2 * (-self.__energies[0] + self.__energies[1]))
                / self.__n_qubits
            )

    def __repr__(self) -> str:
        return "<ExpectationValue field={}>".format(self.__field)
