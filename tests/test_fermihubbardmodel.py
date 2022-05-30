import pytest
import numpy as np
import cirq
import openfermion as of

from fauvqe.models.fermiHubbard import FermiHubbardModel
import fauvqe.utils as utils

@pytest.mark.parametrize(
    "x_dimension,y_dimension,tunneling,coulomb,hamiltonian_options",
    [
        (1,1,0,0,{}),
        (5,5,0,0,{}),
        (5,5,1,1,{}),
        (5,5,-1,1,{}),
        (5,5,1,-1,{}),
        (5,5,1,1,{"chemical_potential":0.0}),
        (1,2,1,1,{"chemical_potential":-1.0,"spinless":False}),
        (1,2,1,1,{"chemical_potential":1.0,"magnetic_field":0.0,"periodic":True,"spinless":False}),
        (1,2,1,1,{"chemical_potential":0.0,"magnetic_field":1.0,"periodic":False,"spinless":True}),
    ]
)

def test_create(x_dimension,y_dimension,tunneling,coulomb,hamiltonian_options):
    fermi_hubbard_hamiltonian = FermiHubbardModel(x_dimension,
                                                y_dimension,
                                                tunneling,
                                                coulomb,
                                                hamiltonian_options)
@pytest.mark.parametrize(
    "transform_name",
    [
        ("jordan_wigner"),
        ("bravyi_kitaev"),
    ]
)
def test_encode(transform_name):
    fermi_hubbard_hamiltonian = FermiHubbardModel(3,3,1,1)
    hamiltonian = fermi_hubbard_hamiltonian._encode_hamiltonian(transform_name)

def test_encode_wrong():
    fermi_hubbard_hamiltonian = FermiHubbardModel(3,3,1,1)
    with pytest.raises(KeyError):
        hamiltonian = fermi_hubbard_hamiltonian._encode_hamiltonian("wrong_encoding")

def test_initial_state():
    Nx=4
    Ny=4
    for dimx in range(1,Nx):
        for dimy in range(1,Ny):
            for Nf in range(Nx*Ny*2):
                fermi_hubbard = FermiHubbardModel(dimx,dimy,1,2)
                fermi_hubbard.set_initial_state_circuit(name="slater",initial_state=[],Nf=Nf)
                result = fermi_hubbard.evaluate(fermi_hubbard.non_interacting_hamiltonian)
                orbital_energies,unitary_rows=fermi_hubbard.diagonalize_non_interacting_hamiltonian()
                expectation = np.real(result)[0]
                ground_truth = sum(list(sorted(orbital_energies)[:Nf]))
                print(expectation,ground_truth)
                assert np.isclose(expectation,ground_truth)

    

def test_initial_state_wrong():
    fermi_hubbard_hamiltonian = FermiHubbardModel(3,3,1,1)
    with pytest.raises(NameError):
        hamiltonian = fermi_hubbard_hamiltonian._get_initial_state("wrong_inital_state",initial_state=None,Nf=None)


