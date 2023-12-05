from cirq import fidelity as cirq_fidelity
from fauvqe.models.qubitModel import QubitModel
from fauvqe.utilities import ketbra
import numpy as np

# ok how do I do this


def energy(model: QubitModel, state: np.ndarray) -> float:
    return model.expectation(state)


def fidelity(target_state: np.ndarray, state: np.ndarray) -> float:
    qid_shape = (2,) * int(np.log2(len(target_state)))
    return cirq_fidelity(target_state, state, qid_shape=qid_shape)


def infidelity(target_state: np.ndarray, state: np.ndarray) -> float:
    return 1 - fidelity(target_state, state)


def trace_distance(target_state: np.ndarray, state: np.ndarray):
    rho1 = ketbra(target_state)
    rho2 = ketbra(state)
    return 0.5 * np.trace(np.abs(rho1 - rho2))
