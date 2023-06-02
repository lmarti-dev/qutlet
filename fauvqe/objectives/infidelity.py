from numpy import float64 as np_float64
from numpy import ndarray as np_ndarray

from fauvqe.objectives.fidelity import Fidelity

class Infidelity(Fidelity):
    def evaluate(self, 
                 wavefunction: np_ndarray,
                 target_state: np_ndarray = None) -> np_float64:
        return 1 - super().evaluate(wavefunction=wavefunction, target_state=target_state)
    