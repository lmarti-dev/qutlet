from tokenize import Name
import cirq
import openfermion as of
import sympy
from fauvqe.models.fockModel import FockModel
import fauvqe.utils as utils
"""
HAMILTONIAN VARIATIONAL ANSATZ
This file contains helper functions related to Fock models VQEs. 
They help implement the Hamiltonian Variational Ansatz (HVA) defined in Phys. Rev. A 92, 042303 (2015). and improved in arXiv:1912.06007

"""

def set_circuit(model: FockModel):

    """
    This function takes a mapping from the encoded fermions to the qubit grid
    if you have

        u11d11  u12d12  u13d13
        u21d21  u22d22  u23d23
        u31d31  u32d32  u33d33

        you have to decide how place your qubits

        openfermion FermionOperator fermi_hubbard() hamiltonians are symbolic representations with the underlying mapping:

        0   1   2   3   4   5     6   7   8   9   10  11    12  13  14  15  16  17
        u11 d11 u12 d12 u13 d13 / u21 d21 u22 d22 u23 d23 / u31 d31 u32 d32 u33 d33

        i.e. the grid is put in a linear list, row-major
        
    """
    if model.encoding_name == "jordan_wigner":
        ps = model.hamiltonian
        assert type(ps) == cirq.PauliSum
        for term in ps:
            print("term: ", term)
            assert type(term) == cirq.PauliString
            
    else:
        raise NameError("No hva implementation found for this encoding: {}".format(model.encoding_name))        


def qubits_shape(qubits):
    last_qubit = max(qubits)
    if isinstance(last_qubit,cirq.LineQubit):
        return last_qubit.x
    elif isinstance(last_qubit,cirq.GridQubit):
        return (last_qubit.row,last_qubit.col)

def fswap_column(model,index,left=True):
    (dimx,dimy) = qubits_shape(model.flattened_qubits)

    if left and index < 1:
        raise ValueError("Cannot fswap with left column when index is {}".format(index))
    if not left and index > dimy-2:
        raise ValueError("Cannot fswap with right column when index is {}".format(index))
    
    a=-1 if left else 1
    op_tree = []

    for x in dimx:
        op_tree+=[of.FSWAP.on(model.qubits[x][index],model.qubits[x][index+a*1])]
    return op_tree

def fswap_all_columns(model,even=True):
    (_,dimy) = qubits_shape(model.flattened_qubits)
    op_tree=[]
    if even:
        a=1
    else:
        a=2
    for index in range(a,dimy,2):
        op_tree += fswap_column(model=model,index=index,left=True)
    model.circuit.append(op_tree)

def onsite_layer(model):
    pass

def optimize_mapping(self):
    # need to check whether there is an optimal mappng of fermions to qubits
    # it seems pretty straightforward
    pass