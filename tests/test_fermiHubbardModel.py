from curses import has_il
from mimetypes import init
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
    fermi_hubbard_hamiltonian.encoding_name = transform_name
    hamiltonian = fermi_hubbard_hamiltonian._encode_fock_hamiltonian()

def test_encode_wrong():
    fermi_hubbard_hamiltonian = FermiHubbardModel(3,3,1,1)
    with pytest.raises(KeyError):
        fermi_hubbard_hamiltonian.encoding_name = "wrong_encoding"
        hamiltonian = fermi_hubbard_hamiltonian._encode_fock_hamiltonian()

initial_state_data = [{"Nx":Nx,"Ny":Ny,"Nf":Nf,"qubit_maps":qubit_maps} 
                        for Nx in range(1,3)
                        for Ny in range(1,3)
                        for Nf in range(Nx*Ny*2)
                        for qubit_maps in [None,(utils.flip_cross_rows,),(utils.flip_cross_rows,utils.alternating_indices_to_sectors)]
                    ]
@pytest.mark.parametrize('initial_state_data', initial_state_data)
def test_initial_state(initial_state_data):
    initial_state(**initial_state_data)

def initial_state(Nx,Ny,Nf,qubit_maps,**kwargs):
    fermi_hubbard = FermiHubbardModel(x_dimension=Nx,
                                    y_dimension=Ny,
                                    tunneling=1,
                                    coulomb=2,
                                    hamiltonian_options={"chemical_potential":0.0},
                                    qubit_maps=qubit_maps)
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
        fermi_hubbard_hamiltonian._get_initial_state("wrong_inital_state",initial_state=None,Nf=None)


def test_diagonalize_non_interacting_hamiltonian():
    pass

