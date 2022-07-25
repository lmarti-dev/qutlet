"""
    Test class ANNNI(SpinModelFC)
"""
# external import
import cirq
import pytest
import numpy as np


# internal import
from fauvqe import ANNNI, Converter


@pytest.mark.parametrize(
    "n, J, k, h, boundaries",
    [
        (
            2,
            1,
            1,
            1,
            0,
        ),
    ]
)
def test_constructor(n, J, k, h, boundaries):
    annni_obj = ANNNI(n, J, k, h, boundaries)

    assert annni_obj.qubitype == "GridQubit"
    assert len(annni_obj.n) == 2


def test_converter():
    pass


#############################################
#                                           #
#               Test Asserts                #
#                                           #
#############################################
@pytest.mark.parametrize(
    "n, J, k, h, boundaries",
    [
        (
            2,
            1,
            1,
            1,
            None,
        ),
    ]
)
def test_constructor_errors(n, J, k, h, boundaries):
    with pytest.raises(AssertionError):
        annni_obj = ANNNI(n, J, k, h, boundaries)