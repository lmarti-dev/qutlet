from cirq import fidelity as cirq_fidelity
from qutlet.models import QubitModel
from qutlet.utilities import ketbra
import numpy as np

"""These objective are destined for statevector simulation and must be used with a statevector simulator
"""


def energy_objective(model: QubitModel) -> callable:
    def fun(state: np.ndarray):
        return np.real(model.statevector_expectation(state))

    return fun


def fidelity_objective(model: QubitModel, target_state: np.ndarray) -> callable:
    def fun(state: np.ndarray):
        return np.real(cirq_fidelity(target_state, state, qid_shape=model.qid_shape))

    return fun


def infidelity_objective(
    model: QubitModel,
    target_state: np.ndarray,
) -> callable:
    def fun(state: np.ndarray):
        return np.real(1 - fidelity_objective(model, target_state, state))

    return fun


def trace_distance_objective(target_state: np.ndarray) -> callable:
    def fun(state: np.ndarray):
        rho1 = ketbra(target_state)
        rho2 = ketbra(state)
        return np.real(0.5 * np.trace(np.abs(rho1 - rho2)))

    return fun
