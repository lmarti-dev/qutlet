import pytest
import json

import fauvqe.json
from fauvqe import ADAM


@pytest.mark.parametrize(
    "c",
    [
        [0],
        [1.1],
        [1.1j],
        [1 + 1j],
    ],
)
def test_complex(c):
    s = fauvqe.json.dumps(c)

    assert fauvqe.json.loads(s) == c


def test_restore_error_unknown():
    s = json.dumps(
        {
            fauvqe.json.JSONEncoder.FLAG: fauvqe.json.JSONEncoder.FLAG_RESTORABLE,
            fauvqe.json.JSONEncoder.RESTORABLE_TYPE: "I_DO_NOT_EXIST",
            fauvqe.json.JSONEncoder.RESTORABLE_DATA: {},
        }
    )

    with pytest.raises(NotImplementedError):
        fauvqe.json.loads(s)


def test_optimiser():
    s = json.dumps(
        {
            fauvqe.json.JSONEncoder.FLAG: fauvqe.json.JSONEncoder.FLAG_RESTORABLE,
            fauvqe.json.JSONEncoder.RESTORABLE_TYPE: "ADAM",
            fauvqe.json.JSONEncoder.RESTORABLE_DATA: {"constructor_params": {}},
        }
    )
    adam = fauvqe.json.loads(s)

    assert isinstance(adam, ADAM)
