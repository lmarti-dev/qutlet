"""
    Use QAOA Ising to test if ADAM optimiser works
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
from fauvqe import Ising, Optimiser, ADAM, ExpectationValue


def test_set_optimiser():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_circuit("qaoa", 1)
    adam = ADAM()
    objective = ExpectationValue(ising)
    adam.optimise(objective)


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
    eps = 10 ** -3
    exp_val_z = ExpectationValue(ising_obj, field="Z")
    adam = ADAM(
        eps=eps,
        break_param=25,
        a=4 * 10 ** -2,
    )
    adam.optimise(exp_val_z)

    wf = ising_obj.simulator.simulate(
        ising_obj.circuit,
        param_resolver=ising_obj.get_param_resolver(ising_obj.circuit_param_values),
    ).state_vector()
    # Result smaller than -0.5 up to eta
    assert -0.5 > exp_val_z.evaluate(wf) - eps
    # Result smaller than -0.5 up to eta


def test_adam_multiple_initialisers():
    ising1 = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
    )
    ising1.set_circuit("qaoa", 2)
    ising1.set_circuit_param_values(0.3 * np.ones(np.size(ising1.circuit_param)))
    ising2 = Ising(
        "GridQubit",
        [1, 2],
        np.ones((0, 2)),
        np.ones((1, 1)),
        np.ones((1, 2)),
    )
    ising2.set_circuit("qaoa", 1)

    adam = ADAM()

    objective1 = ExpectationValue(ising1, field="Z")

    res1 = adam.optimise(objective1)

    objective2 = ExpectationValue(ising2, field="Z")
    res2 = adam.optimise(objective2)

    print(res1, res2)


#############################################################
#                     Test errors                           #
#############################################################
def test_adam_break_cond_assert():
    with pytest.raises(AssertionError):
        ADAM(break_cond="atol")
