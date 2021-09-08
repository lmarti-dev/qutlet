# external imports
import pytest
import numpy as np
import cirq

# internal imports
from fauvqe import Ising, ADAM, ExpectationValue, UtCost, GradientOptimiser

class MockGradientOptimiser(GradientOptimiser):
    def _cpv_update(self, temp_cpv: np.ndarray, _n_jobs: Integral, step: Integral, indices: Optional[List[int]] = None):
        super()._cpv_update(self, temp_cpv: np.ndarray, _n_jobs: Integral, step: Integral, indices: Optional[List[int]] = None)

        
#############################################################
#                                                           #
#                     Test errors                           #
#                                                           #
#############################################################
def test_abstract_gradient_optimiser():
    with pytest.raises(TypeError):
        GradientOptimiser()

