import pytest
import numpy as np
import cirq
import openfermion as of

from qutlet.models.fermiHubbardModel import FermiHubbardModel

from qutlet.utilities.circuit import populate_empty_qubits


@pytest.mark.parametrize(
    "x_dimension,y_dimension,tunneling,coulomb,hamiltonian_options",
    [
        (1, 1, 0, 0, {}),
        (5, 5, 0, 0, {}),
        (5, 5, 1, 1, {}),
        (5, 5, -1, 1, {}),
        (5, 5, 1, -1, {}),
        (5, 5, 1, 1, {"chemical_potential": 0.0}),
        (1, 2, 1, 1, {"chemical_potential": -1.0, "spinless": False}),
        (
            1,
            2,
            1,
            1,
            {
                "chemical_potential": 1.0,
                "magnetic_field": 0.0,
                "periodic": True,
                "spinless": False,
            },
        ),
        (
            1,
            2,
            1,
            1,
            {
                "chemical_potential": 0.0,
                "magnetic_field": 1.0,
                "periodic": False,
                "spinless": True,
            },
        ),
    ],
)
def test_create(x_dimension, y_dimension, tunneling, coulomb, hamiltonian_options):
    fermi_hubbard_hamiltonian = FermiHubbardModel(
        x_dimension=x_dimension,
        y_dimension=y_dimension,
        tunneling=tunneling,
        coulomb=coulomb,
        hamiltonian_options=hamiltonian_options,
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
        x_dimension=3,
        y_dimension=3,
        tunneling=1,
        coulomb=1,
        encoding_options={"encoding_name": transform_name},
    )
    hamiltonian = fermi_hubbard_hamiltonian._encode_fock_hamiltonian()


def test_encode_wrong():
    fermi_hubbard_hamiltonian = FermiHubbardModel(
        x_dimension=3, y_dimension=3, tunneling=1, coulomb=1
    )
    with pytest.raises(KeyError):
        fermi_hubbard_hamiltonian.encoding_options["encoding_name"] = "wrong_encoding"
        fermi_hubbard_hamiltonian._encode_fock_hamiltonian()


def test_initial_state_wrong():
    fermi_hubbard_hamiltonian = FermiHubbardModel(
        x_dimension=3, y_dimension=3, tunneling=1, coulomb=1
    )
    with pytest.raises(NameError):
        fermi_hubbard_hamiltonian._get_initial_state_circuit(
            "wrong_inital_state", initial_state=None, system_fermions=None
        )


def test_diagonalize_non_interacting_hamiltonian():
    pass
