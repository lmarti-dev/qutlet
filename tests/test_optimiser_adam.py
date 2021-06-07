"""
    Use QAOA Ising to test if ADAM optimiser works
    1.create Ising object + simple 4 qubit QAOA?
    2. set_optimiser
    3.ising.optimise()

    Later: possibly put into test class 

    26.03.2020:
        test_optimse currently fails as energy() is jZZ-hX energy in optimiser
        but one needs to give in this case the optimiser energy(.., field='Z')
        Needs to be fixed by generalisation of Ising.set_simulator()

    08.04.21: Need to add some results/run time test
"""
# external imports
import pytest
import numpy as np

# internal imports
from fauvqe import Ising, ADAM, ExpectationValue


def test_set_optimiser():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_circuit("qaoa", 1)
    adam = ADAM()
    objective = ExpectationValue(ising)
    adam.optimise(objective, n_jobs=1)


# This is potentially a higher effort test:
#############################################################
#                                                           #
#                  Sequential version                       #
#                                                           #
#############################################################
@pytest.mark.higheffort
def test_optimise():
    ising = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
    )
    ising.set_circuit("qaoa", 2)
    ising.set_circuit_param_values(0.3 * np.ones(np.size(ising.circuit_param)))
    eps = 10 ** -3
    exp_val_z = ExpectationValue(ising, field="Z")
    adam = ADAM(
        eps=eps,
        break_param=25,
        a=4 * 10 ** -2,
    )
    res = adam.optimise(exp_val_z, n_jobs=1)

    wf = ising.simulator.simulate(
        ising.circuit,
        param_resolver=ising.get_param_resolver(ising.circuit_param_values),
    ).state_vector()
    # Result smaller than -0.5 up to eta
    assert -0.5 > res.get_latest_objective_value() - eps
    # Result smaller than -0.5 up to eta


@pytest.mark.higheffort
def test_adam_multiple_models_and_auto_joblib():
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

    res1 = adam.optimise(objective1, n_jobs=-1)

    objective2 = ExpectationValue(ising2, field="Z")
    res2 = adam.optimise(objective2, n_jobs=-1)

    print(res1, res2)


#############################################################
#                                                           #
#                    Joblib version                         #
#                                                           #
#############################################################
@pytest.mark.higheffort
def test_optimise_joblib():
    ising = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
    )
    ising.set_circuit("qaoa", 2)
    ising.set_circuit_param_values(0.3 * np.ones(np.size(ising.circuit_param)))
    adam = ADAM(
        break_param=25,
        a=4e-2,
    )
    expval_z = ExpectationValue(ising, field="Z")

    res = adam.optimise(expval_z, n_jobs=-1)
    wavefunction = expval_z.simulate(
        param_resolver=ising.get_param_resolver(res.get_latest_step().params)
    )

    # Result smaller than -0.5 up to eta
    assert -0.5 > expval_z.evaluate(wavefunction) - adam._eps
    # Result smaller than -0.5 up to eta

def test_optimise_no_simulator_change():
    ising = Ising(
        "GridQubit", [2, 2], 0.1 * np.ones((1, 2)), 0.5 * np.ones((2, 1)), 0.2 * np.ones((2, 2))
    )
    ising.set_circuit("qaoa", 2)
    ising.set_circuit_param_values(0.3 * np.ones(np.size(ising.circuit_param)))
    ising.set_simulator(simulator_name = "cirq")
    
    adam = ADAM(
        break_param=1,
        a=4e-2,
    )
    expval_z = ExpectationValue(ising, field="Z")
    #assert(ising.simulator == 0)

    res = adam.optimise(expval_z, n_jobs=-1)
    assert(ising.simulator == adam._objective.model.simulator)

def test__get_single_energy():
    ising = Ising(
        "GridQubit", [2, 2], 0.1 * np.ones((1, 2)), 0.5 * np.ones((2, 1)), 0.2 * np.ones((2, 2))
    )
    ising.set_circuit("qaoa", 2)
    ising.set_circuit_param_values(0.3 * np.ones(np.size(ising.circuit_param)))

    adam = ADAM(break_param=1,a=4e-2)
    expval_z = ExpectationValue(ising, field="Z")
    res = adam.optimise(expval_z, n_jobs=-1)
    gg_gradients = adam._get_gradients(adam._objective.model.circuit_param_values, 8)

    # 2 layer, 2 parameters, 2 energies each
    single_energies = np.zeros(2*2*2)
    for j in range(8):
        single_energies[j] = adam._get_single_energy(
            {**{str(adam._circuit_param[i]): adam._objective.model.circuit_param_values[i] for i in range(adam._n_param)}} , 
            j)
    single_energies = np.array(single_energies).reshape((adam._n_param, 2)) 
    se_gradients = np.matmul(single_energies, np.array((1, -1))) / (2 * adam._eps) 
    np.testing.assert_allclose(gg_gradients    , se_gradients, rtol=1e-15, atol=1e-15)


#############################################################
#                                                           #
#                     Test errors                           #
#                                                           #
#############################################################
def test_adam_break_cond_assert():
    with pytest.raises(AssertionError):
        ADAM(break_cond="atol")
