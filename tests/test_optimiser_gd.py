"""
    Use QAOA Ising to test if gradient descent optimiser works
    1.create Ising object + simple 4 qubit QAOA?
    2. set_optimiser
    3.ising_obj.optimise()

    Later: possibly put into test class 

    26.03.2020:
        test_optimse currently fails as energy() is jZZ-hX energy in optimiser
        but one needs to give in this case the optimiser energy(.., field='Z')
        Needs to be fixed by generalisation of Ising.set_simulator()

"""
# external imports
import pytest
import numpy as np

# internal imports
from fauvqe import Ising, Optimiser, GradientDescent, ExpectationValue


def test_set_optimiser():
    ising_obj = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising_obj.set_circuit("qaoa", {"p": 1})
    gd = GradientDescent()
    obj = ExpectationValue(ising_obj)
    gd.optimise(obj)


# This is potentially a higher effort test:
@pytest.mark.higheffort
def test_optimise():
    ising_obj = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
        "Z"
    )
    ising_obj.set_circuit("qaoa", {"p": 2, "H_layer": False})
    ising_obj.set_circuit_param_values(0.3 * np.ones(np.size(ising_obj.circuit_param)))
    eta = 2e-2
    gd = GradientDescent(
        break_param=25,
        eta=eta,
    )
    obj = ExpectationValue(ising_obj)
    res = gd.optimise(obj)

    final_step = res.get_latest_step()

    assert -0.5 > final_step.objective - eta
    # Result smaller than -0.5 up to eta


#############################################################
#                     Test errors                           #
#############################################################
def test_GradientDescent_break_cond_assert():
    with pytest.raises(AssertionError):
        GradientDescent(break_cond="atol")


def test_optimiser_optimise_assert():
    # Test if abstract base class can be initiated
    with pytest.raises(TypeError):
        Optimiser()
