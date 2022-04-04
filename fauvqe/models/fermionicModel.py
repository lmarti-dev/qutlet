from __future__ import annotations

# External import

import openfermion as of

# Internal import
from fauvqe.models.fockModel import FockModel


class FermionicModel(FockModel):

    r"""
    Fock model subclass that implements fermionic operators, 
    i.e creators and annihilators which follow anticommutation rules.
    this class also implements the encoding (and not the fockmodel class),
    because the encodings are dependant on fermionic properties

    """
    def __init__(self,
                qubittype,
                n):
        super().__init__(qubittype, n)

    def _encode_hamiltonian(self,transform_name: str) -> of.QubitOperator:
        """
        use an openfermion transform to encode the occupation basis hamiltonian
        into a qubit hamiltonian. choices are the ones imported on line 5
        """
        transforms_dict=dict()
        transforms_dict["jordan_wigner"]=of.jordan_wigner
        transforms_dict["bravyi_kitaev"]=of.bravyi_kitaev
        
        try:
            return transforms_dict[transform_name](self.fock_hamiltonian)
        except KeyError:
            raise KeyError("No transform named {}. Allowed transforms: {}".format(transform_name,transforms_dict.keys()))

    def set_circuit(self,
                    qalgorithm: str
                    ):
        # set fermions on qubits -> initial states -> h/v hopping - fswap loop -> onsite 
        if qalgorithm == "hva":
            pass



