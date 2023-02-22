import pytest
import openfermion as of
from fauvqe.models.fermionicModel import FermionicModel

def test_init():
    fm = FermionicModel(qubittype="GridQubit",n=10,encoding_name="jordan_wigner")

def test_set_hamiltonian():
    pass

def test_map_qubits():
    pass

def test_remap_fermion_hamiltonian():
    fh1 = sum((*(of.FermionOperator('{}^ {}'.format(x,x+1),.5) for x in range(10)),
                of.FermionOperator("3",2),
                of.FermionOperator("13^",5),
                of.FermionOperator("1^ 2^ 3 4",1.2),
                of.FermionOperator("1^ 9 9^",.5))
                )

    fh2 = sum((*(of.FermionOperator('{}^ {}'.format(x+1,x+2),.5) for x in range(10)),
                of.FermionOperator("4",2),
                of.FermionOperator("14^",5),
                of.FermionOperator("2^ 3^ 4 5",1.2),
                of.FermionOperator("2^ 10 10^",.5))
                )

    def fock_map(x):
        return x+1 
    fh1_remap=FermionicModel.remap_fermion_hamiltonian(fock_hamiltonian=fh1,fock_map=fock_map)
    assert fh1_remap == fh2

def test_alternating_to_sectors_remap():
    fh = of.fermi_hubbard(3,3,1,1)
    FermionicModel.remap_fermion_hamiltonian(fh)
