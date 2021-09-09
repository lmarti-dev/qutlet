# external imports
import pytest
import numpy as np
import cirq

from numbers import Real, Integral
from typing import Literal, Optional, Dict, List
# internal imports
from fauvqe import Ising, ADAM, ExpectationValue, UtCost, GradientOptimiser, Optimiser

class MockGradientOptimiser(GradientOptimiser):
    def _optimise_step(self, temp_cpv: np.ndarray, _n_jobs: Integral, step: Integral):
        super()._optimise_step(temp_cpv, _n_jobs, step)

#############################################################
#                                                           #
#                     Test errors                           #
#                                                           #
#############################################################
def test_abstract_gradient_optimiser():
    with pytest.raises(TypeError):
        GradientOptimiser()

def test_optimiser_optimise_assert():
    # Test if abstract base class can be initiated
    with pytest.raises(TypeError):
        Optimiser()
        
def test_parameter_update_NotImplemented():
    with pytest.raises(NotImplementedError):
        mockopt = MockGradientOptimiser()
        mockopt._optimise_step(np.ones(2), 1, 1)
