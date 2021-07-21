import pathlib
import tempfile

import numpy as np
import pytest

from fauvqe import Ising, ADAM, ExpectationValue, OptimisationResult, CVaR


def get_simple_result():
    ising = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
    )
    ising.set_circuit("qaoa", {"p":1})
    adam = ADAM()
    objective = ExpectationValue(ising)

    return adam.optimise(objective, n_jobs=1)


def test_store_and_restore_ising():
    res = get_simple_result()
    temp_path = pathlib.Path(tempfile.gettempdir()) / "fauvqe-pytest.json"

    res.store(temp_path, overwrite=True)

    print(res)
    print(OptimisationResult.restore(temp_path))

    assert res == OptimisationResult.restore(temp_path)

    temp_path.unlink()


def test_no_overwrite():
    temp_path = pathlib.Path(tempfile.gettempdir()) / "fauvqe-pytest.json"
    temp_path.touch()
    res = get_simple_result()

    with pytest.raises(FileExistsError):
        res.store(temp_path)

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
    adam = ADAM(break_cond="iterations", break_param=3)
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
    adam = ADAM(break_cond="iterations", break_param=5)
    objective = ExpectationValue(ising)
    res1 = adam.optimise(objective, n_jobs=1)

    assert len(res1.get_steps()) == 5

    adam = ADAM(break_cond="iterations", break_param=10)
    res2 = adam.optimise(objective, n_jobs=1, continue_at=res1)

    assert len(res2.get_steps()) == 10

    steps1 = res1.get_steps()
    steps2 = res2.get_steps()

    for i in range(len(steps1)):
        np.testing.assert_equal(steps2[i].params, steps1[i].params)
