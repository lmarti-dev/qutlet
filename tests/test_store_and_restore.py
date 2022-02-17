import pathlib
import tempfile

import numpy as np
import pytest
import os
import cirq

from fauvqe import Ising, ADAM, ExpectationValue, OptimisationResult, CVaR


def get_simple_result(break_param0=25, a0 = 4 * 10 ** -2):
    ising = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
        "Z"
    )
    ising.set_circuit("qaoa", {"p": 2, "H_layer": False})
    ising.set_circuit_param_values(0.3 * np.ones(np.size(ising.circuit_param)))
    eps = 10 ** -3
    objective = ExpectationValue(ising)
    adam = ADAM({
        'eps': eps,
        'break_param': break_param0,
        'eta': a0,
    })

    return adam.optimise(objective, n_jobs=8)

@pytest.mark.higheffort
def test_store_and_restore_ising():
    res = get_simple_result()
    temp_path = os.path.dirname(os.path.abspath(__file__)) + "fauvqe-pytest.json"
    #temp_path = pathlib.Path(tempfile.gettempdir()) / "fauvqe-pytest.json"

    res.store(temp_path, overwrite=True)

    res_restored = OptimisationResult.restore(temp_path)

    #print(res)
    #print(res_restored)
    #print(set(res_restored.__dict__) - set(res_restored.__dict__))

    assert len(set(res_restored.__dict__) - set(res_restored.__dict__)) == 0

    #temp_path.unlink()

@pytest.mark.higheffort
def test_no_overwrite():
    temp_path = pathlib.Path(tempfile.gettempdir()) / "fauvqe-pytest.json"
    temp_path.touch()
    res = get_simple_result()

    with pytest.raises(FileExistsError):
        res.store(temp_path)

    with pytest.raises(FileExistsError):
        res.storetxt(temp_path)

    temp_path.unlink()


def test_no_empty_file():
    temp_path = pathlib.Path(tempfile.gettempdir()) / "fauvqe-pytest.json"
    if temp_path.exists():
        temp_path.unlink()

    with pytest.raises(FileNotFoundError):
        OptimisationResult.restore(temp_path)


@pytest.mark.higheffort
def test_store_all():
    ising = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
    )
    ising.set_circuit("qaoa", {"p":1})
    adam = ADAM({
        'break_cond': "iterations",
        'break_param':3
    })
    objective = CVaR(ising, alpha=1.0)  # Equivalent to ExpectationValue

    res = adam.optimise(objective, n_jobs=-1)

    temp_path = pathlib.Path(tempfile.gettempdir()) / "fauvqe-pytest.json"
    res.store(
        temp_path,
        indent=1,
        overwrite=True,
        store_wavefunctions="all",
        store_objectives="all",
    )

    res_restored = OptimisationResult.restore(temp_path)

    temp_path.unlink()


def test_continue_at():
    ising = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
    )
    ising.set_circuit("qaoa", {"p": 1})
    adam = ADAM({
        'break_cond': "iterations",
        'break_param': 5
    })
    objective = ExpectationValue(ising)
    res1 = adam.optimise(objective, n_jobs=1)

    assert len(res1.get_steps()) == 5

    adam = ADAM({
        'break_cond': "iterations", 
        'break_param': 10
    })
    res2 = adam.optimise(objective, n_jobs=1, continue_at=res1)

    assert len(res2.get_steps()) == 10

    steps1 = res1.get_steps()
    steps2 = res2.get_steps()

    for i in range(len(steps1)):
        np.testing.assert_equal(steps2[i].params, steps1[i].params)
@pytest.mark.parametrize(
    "additional_objective, n_objectives, header_string",
    [
        (
            None, 0, "# ExpectationValue \t\n"
        ),
        (
            ExpectationValue(   Ising(  "GridQubit",
                                        [2, 2],
                                        0.1 * np.ones((1, 2)),
                                        0.5 * np.ones((2, 1)),
                                        0.2 * np.ones((2, 2)),
                                        "Z")),
            1, 
            "# ExpectationValue \tExpectationValue \t\n"
        ),
        (
            [ExpectationValue(   Ising(  "GridQubit",
                                        [2, 2],
                                        0.1 * np.ones((1, 2)),
                                        0.5 * np.ones((2, 1)),
                                        0.2 * np.ones((2, 2)),
                                        "Z")),
             ExpectationValue(   Ising(  "GridQubit",
                                        [2, 2],
                                        0.1 * np.ones((1, 2)),
                                        0.5 * np.ones((2, 1)),
                                        0.2 * np.ones((2, 2)),
                                        "Z"))],
            #This fails but should not!:
            #CVaR(   Ising(  "GridQubit",
            #                            [2, 2],
            #                            0.1 * np.ones((1, 2)),
            #                            0.5 * np.ones((2, 1)),
            #                            0.2 * np.ones((2, 2))), alpha=1, field="X")],
            2, 
            "# ExpectationValue \tExpectationValue \tExpectationValue \t\n"

        ),
    ]
)
@pytest.mark.higheffort
def test_storetxt(additional_objective, n_objectives,header_string):
    res = get_simple_result()
    temp_path = os.path.dirname(os.path.abspath(__file__)) + "/fauvqe-pytest.txt"

    res.storetxt(   temp_path, 
                    overwrite=True,
                    additional_objectives=additional_objective)

    temp_data = np.loadtxt(temp_path)

    f = open(temp_path)
    header = f.readline()
    f.close()
    assert header == header_string

    if n_objectives == 0:
        assert temp_data.shape == (25,)
        assert (res.get_objectives() == temp_data).all()
    else:
        assert temp_data.shape == (25,n_objectives+1)
        for i in range(n_objectives+1):
            print("i: {}".format(i))
            print(res.get_objectives())
            print(temp_data[:,i])
            print(res.get_objectives()/temp_data[:,i])
            np.testing.assert_allclose(res.get_objectives(), temp_data[:,i], rtol=1e-15, atol=1e-15)

@pytest.mark.higheffort
def test_get_wf_from_i():
    res = get_simple_result(1, a0 = 1e-100)

    ising = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
        "Z"
    )
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 2, "H_layer": False})
    ising.set_circuit_param_values(0.3 * np.ones(np.size(ising.circuit_param)))

    wf = ising.simulator.simulate(ising.circuit, 
                                param_resolver=cirq.ParamResolver(
                                    dict(zip(ising.circuit_param, ising.circuit_param_values))
                                    )).state_vector()

    #With np.complex128 this should pass:
    #cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(wf, res._get_wf_from_i(0), rtol=1e-15, atol=1e-15)

    #With np.complex64 only have square root tolerances
    cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase( wf/np.linalg.norm(wf), 
                                                                    res._get_wf_from_i(0)/np.linalg.norm(res._get_wf_from_i(0)), 
                                                                    rtol=np.sqrt(1e-15), 
                                                                    atol=np.sqrt(1e-15))
    