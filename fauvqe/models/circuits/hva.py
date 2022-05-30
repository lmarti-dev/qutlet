import cirq
import sympy
"""
HAMILTONIAN VARIATIONAL ANSATZ
This file contains helper functions related to Fock models VQEs. 
They help implement the Hamiltonian Variational Ansatz (HVA) described in 
Wecker, D., Hastings, M. B. & Troyer, M. Progress towards practical quantum variational algorithms. Phys. Rev. A 92, 042303 (2015).

Ideally, the FSWAPs would be automated.

"""

def set_circuit(self):

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
    





def optimize_mapping(self):
    # need to check whether there is an optimal mappng of fermions to qubits
    # it seems pretty straightforward
    pass