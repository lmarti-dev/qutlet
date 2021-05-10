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
import cirq

# internal imports
from fauvqe import Ising
from fauvqe.optimisers import Optimiser, GradientDescent


def test_set_optimiser():
    ising_obj = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising_obj.set_circuit("qaoa", 1)
    ising_obj.set_optimiser(GradientDescent())


# This is potentially a higher effort test:
@pytest.mark.higheffort
def test_optimise():
    ising_obj = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
    )
    ising_obj.set_circuit("qaoa", 2)
    ising_obj.set_circuit_param_values(0.3 * np.ones(np.size(ising_obj.circuit_param)))
    eta = 2e-2
    ising_obj.set_optimiser(
        GradientDescent(
            break_param=25,
            eta=eta,
        ),
        obj_func="Z",
    )
    ising_obj.optimiser.optimise()
    wf = ising_obj.simulator.simulate(
        ising_obj.circuit,
        param_resolver=ising_obj.optimiser._get_param_resolver(ising_obj.circuit_param_values),
    ).state_vector()
    # Result smaller than -0.5 up to eta
    assert -0.5 > ising_obj.energy(wf, field="Z") - eta
    # Result smaller than -0.5 up to eta


@pytest.mark.higheffort
def test_optimise_print():
    ising_obj = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
    )
    ising_obj.set_circuit("qaoa", 2)
    ising_obj.set_circuit_param_values(0.3 * np.ones(np.size(ising_obj.circuit_param)))
    eta = 2e-2
    ising_obj.set_optimiser(
        GradientDescent(
            break_param=25,
            eta=eta,
            n_print=5,
        ),
        obj_func="Z",
    )
    ising_obj.optimiser.optimise()
    wf = ising_obj.simulator.simulate(
        ising_obj.circuit,
        param_resolver=ising_obj.optimiser._get_param_resolver(ising_obj.circuit_param_values),
    ).state_vector()
    # Result smaller than -0.5 up to eta
    assert -0.5 > ising_obj.energy(wf, field="Z") - eta
    # Result smaller than -0.5 up to eta


# Want to make this work but how?
# via ndarray.view() => available for the other objects???
def test_param_view():
    ising_obj = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
    )
    ising_obj.set_circuit("qaoa", 2)
    ising_obj.set_optimiser(GradientDescent())
    # set self.circuit_parm_values to different value and see if pointer works
    # ising_obj.set_circuit_param_values(0.3*np.ones(np.size(ising_obj.circuit_param)) )

    ising_obj.circuit_param_values[0] = 0
    assert (ising_obj.optimiser.circuit_param_values == ising_obj.circuit_param_values).all()


# Need to add some results/run time test

#############################################################
#                     Test errors                           #
#############################################################
def test_GradientDescent_break_cond_assert():
    ising_obj = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising_obj.set_circuit("qaoa", 1)
    ising_obj.set_circuit_param_values(0.314 * np.ones(np.size(ising_obj.circuit_param)))
    ising_obj.set_optimiser(GradientDescent(break_cond="atol"))

    with pytest.raises(AssertionError):
        ising_obj.optimiser.optimise()


def test_optimiser_constructor_assert():
    ising_obj = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising_obj.set_circuit("qaoa", 1)

    with pytest.raises(AssertionError):
        ising_obj.set_optimiser("TestOptimiser")


def test_optimiser_optimise_assert():
    # Test if abstract base class can be initiated
    with pytest.raises(TypeError):
        Optimiser()
