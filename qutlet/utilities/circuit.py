from typing import Tuple
import cirq
import numpy as np
import qutlet.utilities.generic

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # avoids circular dependency issue from lifting utility packages
    from qutlet.models.qubitModel import AbstractModel  # pragma: no cover


def all_pauli_str_commute(psum: cirq.PauliSum) -> bool:
    """check whether all terms in a PauliSum commute with one another
    Args:
        psum (cirq.PauliSum): the Pauli sum to check
    Returns:
        bool: whether they commute
    """

    for pstr1 in psum:
        for pstr2 in psum:
            if not cirq.commutes(pstr1, pstr2):
                return False
    return True


def pauli_str_is_identity(pstr: cirq.PauliString) -> bool:
    """Check whether a Pauli string is the identity, i.e. like IIIIIII
    Args:
        pstr (cirq.PauliString): the pauli string to check
    Raises:
        ValueError: if the input is not a Pauli string
    Returns:
        bool: whether it is the identity
    """
    if not isinstance(pstr, cirq.PauliString):
        raise TypeError("expected PauliString, got: {}".format(type(pstr)))
    return all(
        pstr.gate.pauli_mask
        == np.array([0] * len(pstr.gate.pauli_mask)).astype(np.uint8)
    )


def match_param_values_to_symbols(
    model: "QubitModel", symbols: list, default_value: str = "zeros"
):
    """add values to param_values when some are missing wrt. the param array
    Args:
        model (AbstractModel): the model whose params are to be checked
        symbols (list): the symbols to match
        default_value (str, optional): what to put in the additional params. Defaults to "zeros".
    """
    if model.circuit_param_values is None:
        model.circuit_param_values = np.array([])
    missing_size = np.size(symbols) - np.size(model.circuit_param_values)

    param_default_values = qutlet.utilities.generic.default_value_handler(
        shape=(missing_size,), value=default_value
    )
    if missing_size > 0:
        model.circuit_param_values = np.concatenate(
            (model.circuit_param_values, param_default_values)
        )


def populate_empty_qubits(model: "QubitModel") -> cirq.Circuit:
    """Add I gates to qubits without operations. This is mainly to avoid some errors with measurement in cirq
    Args:
        model (AbstractModel): the model to check
    Returns:
        cirq.Circuit: the circuit with additional I gates
    """
    circuit_qubits = list(model.circuit.all_qubits())
    model_qubits = model.qubits
    missing_qubits = [x for x in model_qubits if x not in circuit_qubits]
    circ = model.circuit.copy()
    if circuit_qubits == []:
        print("The circuit has no qubits")

        circ = cirq.Circuit()
    circ.append([cirq.I(mq) for mq in missing_qubits])
    return circ


def qmap(model: "QubitModel") -> dict:
    """Get a qmap necessary for some openfermion functions
    Args:
        model (AbstractModel): the model we will use to generate the qmap
    Returns:
        dict: the resulting qmap
    """
    flattened_qubits = list(qutlet.utilities.generic.flatten(model.qubits))
    return {k: v for k, v in zip(flattened_qubits, range(len(flattened_qubits)))}


def pauli_str_is_hermitian(pstr: cirq.PauliString, anti: bool = False) -> bool:
    """Check whether some Pauli string is hermitian or anti-hermitian
    Args:
        pstr (cirq.PauliString): the pauli string to check
        anti (bool, optional): Whether to check for hermitianity or anti-hermitianity. Defaults to False.
    Returns:
        bool: whether the pauli string is hermitian/anti-hermitian
    """
    if anti:
        return np.conj(pstr.coefficient) == -pstr.coefficient
    else:
        return np.conj(pstr.coefficient) == pstr.coefficient


def make_pauli_str_hermitian(
    pstr: cirq.PauliString, anti: bool = False
) -> cirq.PauliString:
    """Make a
    Args:
        pstr (cirq.PauliString): _description_
        anti (bool, optional): _description_. Defaults to False.
    Returns:
        cirq.PauliString: _description_
    """
    if pauli_str_is_hermitian(pstr=pstr, anti=anti):
        return pstr
    elif pauli_str_is_hermitian(pstr=pstr, anti=not anti):
        return pstr.with_coefficient(1j * pstr.coefficient)
    else:
        if not anti:
            # hermitian A + A* = re(A)
            return pstr.with_coefficient(np.real(pstr.coefficient))
        else:
            # anti-hermitian A - A* = 1j*im(A)
            return pstr.with_coefficient(1j * np.imag(pstr.coefficient))


def make_pauli_sum_hermitian(psum: cirq.PauliSum, anti: bool = False) -> cirq.PauliSum:
    """Make a pauli sum hermitian or anti-hermitian
    Args:
        psum (cirq.PauliSum): the pauli sum input
        anti (bool, optional): whether to make it anti-hermitian. Defaults to False.
    Returns:
        cirq.PauliSum: the hermitian/anti-hermitian pauli sum
    """
    psum_out = sum([make_pauli_str_hermitian(pstr, anti) for pstr in psum])
    return psum_out


def pauli_sum_is_hermitian(psum: cirq.PauliSum, anti: bool = False):
    return all(pauli_str_is_hermitian(pstr=pstr, anti=anti) for pstr in psum)


def get_param_resolver(
    model: "QubitModel", param_values: np.ndarray
) -> cirq.ParamResolver:
    """Get a param resolver for cirq, i.e. put some numerical values in some symbolic items
    Args:
        model (AbstractModel): the model for which we want a param
        param_values (np.ndarray, optional): the values to put in the place of the symbols.
    Returns:
        cirq.ParamResolver: the param resolver
    """
    joined_dict = {
        **{
            str(model.circuit_param[i]): param_values[i]
            for i in range(len(model.circuit_param))
        }
    }
    return cirq.ParamResolver(joined_dict)


def depth(circuit: cirq.Circuit) -> int:
    """Get the depth of a circuit
    We create a new circuit to repack it, since the original circuit whose depth we want to compute has not been optimised.
    Args:
        circuit (cirq.Circuit): the circuit to evaluate
    Returns:
        int: the depth of the circuit
    """
    depth = len(cirq.Circuit(circuit.all_operations()))
    return depth


def qubits_shape(qubits: Tuple[cirq.Qid]) -> tuple:
    """Get the shape of some qubit list
    Args:
        qubits (Tuple[cirq.Qid]): the qubit list
    Returns:
        tuple: the shape of the qubits
    """
    last_qubit = max(qubits)
    if isinstance(last_qubit, cirq.LineQubit):
        return (last_qubit.x + 1, 1)
    elif isinstance(last_qubit, cirq.GridQubit):
        return (last_qubit.row + 1, last_qubit.col + 1)
