from qutlet.models import RandomFermionicModel
import pytest


@pytest.mark.parametrize("nq, no, to", [(3, 2, 1), (4, 4, 2), (8, 3, 2), (6, 2, 2)])
def test_init_rfm(nq, no, to):
    for n_qubits in (3, 6, 8):
        for spin in (True, False):
            _ = RandomFermionicModel(
                nq, no, to, is_spin_conserved=spin, n_electrons="half-filling"
            )
