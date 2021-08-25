# external imports
import pytest
import numpy as np
import cirq

# internal imports
from fauvqe import Ising, ADAM, ExpectationValue, UtCost, GradientOptimiser

#############################################################
#                                                           #
#                     Test errors                           #
#                                                           #
#############################################################
def test_abstract_gradient_optimiser():
    with pytest.raises(TypeError):
        GradientOptimiser()
