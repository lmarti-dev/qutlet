"""
#  TP2 internal VQE class
#   purpose is to write common code in a compatible manner
#
#  try to use type definitions and numpy arrays as good as possible
#
# use this:
# https://quantumai.google/cirq/tutorials/educators/qaoa_ising
# as a starting point.
# Write in class strucutre. Add automated testing. Put to package.
# Then add own ideas and alternative optimisers, ising circuits etc.

"""
from __future__ import annotations

import abc
from typing import Tuple, List, Optional, Iterable, Union

import numpy as np
import sympy
import cirq
import timeit

from fauvqe.utilities import flatten, get_param_resolver


class QubitModel:
    def __init__(self, qubit_shape: Union[Iterable, int]):
        super().__init__()
        if isinstance(qubit_shape, Iterable):
            self.shape = qubit_shape
            self.n_qubits = int(np.prod(qubit_shape))
        elif isinstance(qubit_shape, int):
            self.shape = (1, qubit_shape)
            self.n_qubits = qubit_shape
        else:
            raise TypeError(f"Expected iterable or int, got {type(qubit_shape)}")
        self.hamiltonian: cirq.PauliSum() = None
        self._qubits = cirq.LineQubit.range(self.n_qubits)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.qubits[np.ravel_multi_index(multi_index=idx, dims=self.shape)]
        else:
            return self.qubits[idx]

    @property
    def qubits(self):
        return self._qubits

    @property
    def qmap(self):
        return {val: ind for ind, val in enumerate(self._qubits)}

    @property
    def param_resolver(self):
        return get_param_resolver(model=self, param_values=self.circuit_param_values)

    @property
    def qid_shape(self):
        return (2,) * self.n_qubits

    def set_target_state(self, state: np.ndarray):
        self.target_state = state

    def _set_hamiltonian(self):
        raise NotImplementedError()  # pragma: no cover

    def expectation(self, state_vector: np.ndarray):
        if len(state_vector.shape) == 2:
            return self.hamiltonian.expectation_from_density_matrix(
                state_vector, qubit_map=self.qmap
            )
        elif len(state_vector.shape) == 1:
            return self.hamiltonian.expectation_from_state_vector(state_vector, qubit_map=self.qmap)
        else:
            raise ValueError("state_vector shape mismatch: {}".format(state_vector.shape))
