import pytest

from qutlet.models.fermi_hubbard_model import FermiHubbardModel
from openfermion import fermi_hubbard, normal_ordered


@pytest.mark.parametrize(
    "x_dimension,y_dimension,tunneling,coulomb,hamiltonian_options",
    [
        (1, 1, 0, 0, {}),
        (5, 5, 0, 0, {}),
        (5, 5, 1, 1, {}),
        (5, 5, -1, 1, {}),
        (5, 5, 1, -1, {}),
        (5, 5, 1, 1, {}),
        (1, 2, 1, 1, {"periodic": True}),
        (
            1,
            2,
            1,
            1,
            {
                "periodic": True,
            },
        ),
        (
            1,
            2,
            1,
            1,
            {"periodic": False},
        ),
    ],
)
def test_create(x_dimension, y_dimension, tunneling, coulomb, hamiltonian_options):
    fermi_hubbard_hamiltonian = FermiHubbardModel(  # noqa F841
        lattice_dimensions=(x_dimension, y_dimension),
        n_electrons=[],
        tunneling=tunneling,
        coulomb=coulomb,
        **hamiltonian_options,
    )
    assert normal_ordered(fermi_hubbard_hamiltonian.fock_hamiltonian) == normal_ordered(
        fermi_hubbard(
            x_dimension=x_dimension,
            y_dimension=y_dimension,
            tunneling=tunneling,
            coulomb=coulomb,
            **hamiltonian_options,
        )
    )


@pytest.mark.parametrize(
    "transform_name",
    [
        ("jordan_wigner"),
        ("bravyi_kitaev"),
    ],
)
def test_encode(transform_name):
    fermi_hubbard_hamiltonian = FermiHubbardModel(
        lattice_dimensions=(3, 3),
        n_electrons=[],
        tunneling=1,
        coulomb=1,
        encoding_options={"encoding_name": transform_name},
    )
    hamiltonian = fermi_hubbard_hamiltonian._encode_fock_hamiltonian()


def test_encode_wrong():
    fermi_hubbard_hamiltonian = FermiHubbardModel(
        lattice_dimensions=(3, 3), n_electrons=[2, 2], tunneling=1, coulomb=1
    )
    with pytest.raises(KeyError):
        fermi_hubbard_hamiltonian.encoding_options["encoding_name"] = "wrong_encoding"
        fermi_hubbard_hamiltonian._encode_fock_hamiltonian()


def test_diagonalize_non_interacting_hamiltonian():
    pass
