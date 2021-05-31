import pathlib
import tempfile

import numpy as np

from fauvqe import Ising, ADAM, ExpectationValue, OptimisationResult


def test_store_and_restore_ising():
    ising = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
    )
    ising.set_circuit("qaoa", 1)
    adam = ADAM()
    objective = ExpectationValue(ising)
    res = adam.optimise(objective, n_jobs=1)
    temp_path = pathlib.Path(tempfile.gettempdir()) / "fauvqe-pytest.json"

    res.store(temp_path, overwrite=True)

    OptimisationResult.restore(temp_path)

    temp_path.unlink()
