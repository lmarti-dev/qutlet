from __future__ import annotations
from tokenize import Name

# External import

import openfermion as of
from cirq import PauliSum
import numpy as np

# Internal import
from fauvqe.models.fockModel import FockModel
import fauvqe.utils as utils


class FermionicModel(FockModel):

    r"""
    Fock model subclass that implements fermionic operators, 
    i.e creators and annihilators which follow anticommutation rules.
    this class also implements the encoding (and not the fockmodel class),
    because the encodings are dependant on fermionic properties

    """
    def __init__(self,
                qubittype,
                n,
                transform_name: str,
                qubit_map: str = "default"):
        super().__init__(qubittype, n)
        self.qubit_map = qubit_map
        self.transform_name=transform_name
        self._set_hamiltonian()
        # this is a reminder of how up and down fermions are spread on the grid
        # the default value is the one given by the fermi_hubbard function, ie
        # u d u d 
        

    def _set_hamiltonian(self, reset: bool = True):
        self.hamiltonian = self._encode_hamiltonian(self.transform_name)


    def map_qubits(self, flattened_qubits):
        # for now do nothing, and keep the alternating scheme
        if self.qubit_map == "default":
            return flattened_qubits
        else:
            raise NameError("qubit_map has an invalid value: {}".format(self.qubit_map))

    def _encode_hamiltonian(self,transform_name: str) -> PauliSum:
        return self._encode_fock_hamiltonian(model=self.fock_hamiltonian,transform_name=transform_name)

    def _encode_fock_hamiltonian(self,model,transform_name: str) -> PauliSum:
        """
        use an openfermion transform to encode the occupation basis hamiltonian
        into a qubit hamiltonian. choices are the ones imported on line 5
        """
        transforms_dict=dict()
        transforms_dict["jordan_wigner"]=of.jordan_wigner
        transforms_dict["bravyi_kitaev"]=of.bravyi_kitaev
        
        try:
            # need to specify the flattened_qubits here otherwise some validation methods will fail when evaluating the expectation
            # of the hamiltonian
            # this function is where all the mapping of qubits will happen
            # the encoding only does a symbolic encoding, regardless of the qubit structure
            # when the qubit operator is converted to pauli sum it determines which fermions is associated with each qubit
            return of.qubit_operator_to_pauli_sum(transforms_dict[transform_name](model),qubits=self.map_qubits(self.flattened_qubits))
        except KeyError:
            raise KeyError("No transform named {}. Allowed transforms: {}".format(transform_name,transforms_dict.keys()))

    def set_circuit(self,
                    qalgorithm: str
                    ):
        # set fermions on qubits -> initial states -> h/v hopping - fswap loop -> onsite 
        if qalgorithm == "hva":
            raise NotImplementedError("Hamiltonian Variational Ansatz doesn't exist yet")
        else:
            raise NameError("{} is not a valid circuit".format(qalgorithm))
    
    


