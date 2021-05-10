"""
Expval module docstring
"""

from typing import Literal, Tuple
import numpy as np

from ..isings.initialisers import Initialiser
from .objective import Objective


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

    def __init__(self, field: Literal["Z", "X"] = "Z"):
        super().__init__()
        assert field in [
            "Z",
            "X",
        ], "Bad argument 'field'. Allowed values are ['X', 'Z' (default)], revieced {}".format(
            field
        )

        self.__field: Literal["Z", "X"] = field
        self.__energies: np.ndarray = None
        self.__n_qubits: int = 0

    def initialise(self, obj_value: Tuple[np.ndarray]) -> None:
        # Type checking
        assert isinstance(
            obj_value, np.ndarray
        ), "Bad argument 'obj_value'. Must be an instance of numpy.ndarray"
        self.__energies = obj_value
        self.__n_qubits = np.log2(obj_value.size())

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

        raise NotImplementedError("Unknown field {}".format(self.__field))

    def __repr__(self) -> str:
        return "<ExpectationValue field={}>".format(self.__field)
