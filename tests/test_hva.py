import pytest
import fauvqe.models.fermiHubbard as fh
import fauvqe.models.circuits.hva as hva
import fauvqe.utils as utils
import openfermion as of

def test_init():
    fermi_hubbard=fh.FermiHubbardModel(x_dimension=2,
                                    y_dimension=2,
                                    tunneling=2,
                                    coulomb=2,
                                    fock_maps=(0,2,1,3,4,6,5,7),
                                    Z_snake=(0,1,2,3,7,6,5,4)
                                    )
    hva.set_circuit(fermi_hubbard)



@pytest.mark.parametrize(
    "x_dimension,y_dimension,Z_snake,correct,horizontal,even",
    [
        (2,2,(0,3,4,7,1,2,5,6),((1,2),(4,5),(6,7)),True,True),
        (2,2,(0,3,4,7,1,2,5,6),((0,4),(1,5),(2,6),(3,7)),False,True),
        (2,2,(0,1,2,3,7,6,5,4),((0,1),(2,3),(4,5),(6,7)),True,True),
        (2,2,(0,1,2,3,7,6,5,4),((3,7),),False,True),
        (2,2,(0,1,2,4,7,6,5,3),((0,1),(4,5)),True,True),
        (2,2,(0,1,2,4,7,6,5,3),((1,2),(5,6)),True,False),
        (2,2,(0,1,2,4,7,6,5,3),(((3,7),)),False,False),
    ])

def test_fswap_max(x_dimension,y_dimension,Z_snake,correct,horizontal,even):
    fermi_hubbard=fh.FermiHubbardModel(x_dimension=x_dimension,
                                y_dimension=y_dimension,
                                tunneling=2,
                                coulomb=2,
                                fock_maps=(0,2,1,3,4,6,5,7),
                                Z_snake=Z_snake
                                )
    correct_gates = [of.FSWAP.on(
                    fermi_hubbard.flattened_qubits[q[0]],
                    fermi_hubbard.flattened_qubits[q[1]])
                    for q in correct]
    gates = hva.fswap_max(model=fermi_hubbard,horizontal=horizontal,even=even)
    print(gates)
    print(correct_gates)
    assert utils.lists_have_same_elements(gates,correct_gates)

# test_fswap_max(2,2,(0,3,4,7,1,2,5,6),((1,2),(4,5),(6,7)),True,True)