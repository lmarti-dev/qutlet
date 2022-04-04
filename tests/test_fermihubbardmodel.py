import pytest


from fauvqe.models.fermiHubbard import FermiHubbardModel


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
        hamiltonian = fermi_hubbard_hamiltonian._encode_hamiltonian("obviously_wrong_encoding")

def test_initial_state():
    fermi_hubbard_hamiltonian = FermiHubbardModel(3,3,1,1)
    op_tree=fermi_hubbard_hamiltonian.get_initial_state(name="slater")

    
def test_initial_state_wrong():
    fermi_hubbard_hamiltonian = FermiHubbardModel(3,3,1,1)
    with pytest.raises(NameError):
        hamiltonian = fermi_hubbard_hamiltonian.get_initial_state("obviously_wrong_encoding")
