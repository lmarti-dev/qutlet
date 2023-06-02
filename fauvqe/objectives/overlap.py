from numpy import float64 as np_float64
from numpy import ndarray as np_ndarray
from numpy import sqrt as np_sqrt

from fauvqe.objectives.fidelity import Fidelity

class Overlap(Fidelity):
    def evaluate(self, 
                 wavefunction: np_ndarray,
                 target_state: np_ndarray = None) -> np_float64:
        return np_sqrt(super().evaluate(wavefunction=wavefunction, target_state=target_state))