import pytest
import numpy as np
import cirq
import openfermion as of

from models.fermiHubbardModel import FermiHubbardModel


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
        hamiltonian = fermi_hubbard_hamiltonian._encode_fock_hamiltonian()


initial_state_data = [
    {"Nx": Nx, "Ny": Ny, "Nf": Nf}
    for Nx in range(2, 3)
    for Ny in range(1, 3)
    for Nf in range(Nx * Ny * 2)
]


@pytest.mark.parametrize("initial_state_data", initial_state_data)
def test_initial_state(initial_state_data):
    initial_state(**initial_state_data)


def initial_state(Nx, Ny, Nf):
    fermi_hubbard = FermiHubbardModel(
        x_dimension=Nx,
        y_dimension=Ny,
        tunneling=1,
        coulomb=2,
        hamiltonian_options={"chemical_potential": 0.0, "periodic": False},
    )
    fermi_hubbard.set_initial_state_circuit(name="slater", initial_state=[], Nf=Nf)
    fermi_hubbard.add_missing_qubits()
    result = fermi_hubbard.evaluate(fermi_hubbard.non_interacting_model.hamiltonian)

    expectation = np.real(result)[0]

    # method 1
    # orbital_energies,unitary_rows=fermi_hubbard.diagonalize_non_interacting_hamiltonian()
    # ground_energy = sum(list(sorted(orbital_energies)[:Nf]))

    # method 2
    # IndexError for Nx=1, Ny=1, Nf=1
    sparse_hamiltonian = of.get_sparse_operator(
        fermi_hubbard.non_interacting_model.fock_hamiltonian
    )
    ground_energy, ground_state = of.jw_get_ground_state_at_particle_number(
        sparse_hamiltonian, fermi_hubbard.Nf
    )

    print(f"exp: {expectation}, exact: {ground_energy}")
    print(f"result: {result}")
    print(fermi_hubbard.circuit)
    print(fermi_hubbard.non_interacting_model.hamiltonian)
    assert np.isclose(expectation, ground_energy)


def test_initial_state_wrong():
    fermi_hubbard_hamiltonian = FermiHubbardModel(
        x_dimension=3, y_dimension=3, tunneling=1, coulomb=1
    )
    with pytest.raises(NameError):
        fermi_hubbard_hamiltonian._get_initial_state(
            "wrong_inital_state", initial_state=None, Nf=None
        )


def test_diagonalize_non_interacting_hamiltonian():
    pass
