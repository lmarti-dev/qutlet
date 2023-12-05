import pytest
import numpy as np
from qutlet.optimisers.scipy_optimisers import ScipyOptimisers
from qutlet.objectives.abstractexpectationvalue import AbstractExpectationValue

from qutlet.models.qubitModel import AbstractModel
from typing import Dict, Tuple

import cirq
import sympy


class DummyModel(QubitModel):
    def __init__(self, n_qubits):
        super().__init__(n_qubits)
        self._set_hamiltonian()

    def copy(self) -> AbstractModel:
        pass  # pragma: no cover

    def energy(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def _set_hamiltonian(self):
        self._hamiltonian = sum([cirq.Z(q) for q in self.qubits])

    def set_circuit(self):
        symbols = [sympy.Symbol(str(q)) for q in self.qubits]
        self.circuit_param = symbols
        self.circuit_param_values = np.zeros(len(symbols))
        self.circuit = cirq.Circuit(
            [
                cirq.ry(rads=symbols[ii]).on(self.qubits[ii])
                for ii in range(len(self.qubits))
            ]
        )

    def ground_state(self):
        wf = np.zeros(2 ** len(self.qubits), dtype=np.complex64)
        wf[-1] = 1
        return wf

    def initial_state(self):
        wf = np.zeros(2 ** len(self.qubits), dtype=np.complex64)
        wf[0] = 1
        return wf

    def from_json_dict(self):
        pass

    def to_json_dict(self):
        pass


def test_init():
    minimize_options = {"method": "Powell"}
    method_options = {
        "ftol": 1e-17,
        "xtol": 1e-17,
        "maxfev": 1e10,
        "maxiter": 1e10,
    }
    optimiser = ScipyOptimisers(
        method_options=method_options,
        minimize_options=minimize_options,
    )
    json = optimiser.to_json_dict()

    optimiser2 = optimiser.from_json_dict(json)
    assert optimiser == optimiser2


def test_optimize():
    n_qubits = [1, 3]
    model = DummyModel(n_qubits=n_qubits)
    model.set_circuit()

    objective = AbstractExpectationValue(model=model)
    minimize_options = {"method": "COBYLA"}
    for save_function_calls in (True, False):
        optimiser = ScipyOptimisers(
            minimize_options=minimize_options,
            initial_state=model.initial_state(),
            save_each_function_call=save_function_calls,
        )
        for initial_params in ("zeros", "ones", "random"):
            result = optimiser.optimise(
                objective=objective, initial_params=initial_params
            )
            final_state = np.real(result.get_latest_step().wavefunction())
            ground_state = model.ground_state()
            assert (
                cirq.fidelity(
                    ground_state,
                    final_state,
                    qid_shape=(2,) * len(model.qubits),
                )
                > 0.999
            )
