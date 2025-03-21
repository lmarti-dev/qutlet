import abc
from typing import Iterable, Union

import numpy as np
from cirq import PauliSum, NamedQubit, Qid
from uuid import uuid4


class QubitModel(abc.ABC):
    def __init__(self, qubit_shape: Union[Iterable, int]):
        super().__init__()
        self.qubit_shape = None
        if isinstance(qubit_shape, Iterable):
            self.qubit_shape = qubit_shape
            self.n_qubits = int(np.prod(qubit_shape))
        elif isinstance(qubit_shape, int):
            self.qubit_shape = (1, qubit_shape)
            self.n_qubits = qubit_shape
        else:
            raise TypeError(f"Expected iterable or int, got {type(qubit_shape)}")
        self.hamiltonian: PauliSum = None
        self._sys_name = uuid4()
        self._qubits = [
            NamedQubit(f"{self._sys_name}-qubit-{x}") for x in range(self.n_qubits)
        ]

    def __getitem__(self, idx) -> NamedQubit:
        if isinstance(idx, tuple):
            return self.qubits[
                np.ravel_multi_index(multi_index=idx, dims=self.qubit_shape)
            ]
        else:
            return self.qubits[idx]

    @property
    def qubits(self) -> list[Qid]:
        return self._qubits

    @property
    def qmap(self) -> dict[int, int]:
        return {val: ind for ind, val in enumerate(self._qubits)}

    @property
    def qid_shape(self) -> tuple:
        return (2,) * self.n_qubits

    @property
    def hamiltonian_matrix(self) -> np.ndarray:
        return self.hamiltonian.matrix(self.qubits)

    @abc.abstractmethod
    def _set_hamiltonian(self):
        raise NotImplementedError()  # pragma: no cover

    @abc.abstractmethod
    def __to_json__(self):
        raise NotImplementedError()  # pragma: no cover

    def statevector_expectation(self, state_vector: np.ndarray) -> float:
        if len(state_vector.shape) == 2:
            return np.real(
                self.hamiltonian.expectation_from_density_matrix(
                    state_vector, qubit_map=self.qmap
                )
            )
        elif len(state_vector.shape) == 1:
            return np.real(
                self.hamiltonian.expectation_from_state_vector(
                    state_vector, qubit_map=self.qmap
                )
            )
        else:
            raise ValueError(
                "state_vector shape mismatch: {}".format(state_vector.shape)
            )
