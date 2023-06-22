from typing import Callable, Dict, Tuple, Union, Sequence, Optional
import openfermion as of
import cirq
import copy

from fauvqe.models.fermionicModel import FermionicModel
from fauvqe.utilities.generic import flatten, index_bits


class FermionOperatorModel(FermionicModel):
    def __init__(
        self,
        fermion_operator: of.FermionOperator,
        encoding_options: dict = None,
        n: tuple = None,
        **kwargs
    ):
        if not isinstance(fermion_operator, of.FermionOperator):
            raise TypeError(
                "Expected a FermionOperator, got: {}".format(type(fermion_operator))
            )
        self.fermion_operator = fermion_operator
        if n is None:
            n = (1, of.count_qubits(operator=self.fermion_operator))
        if encoding_options is None:
            encoding_options = {"encoding_name": "jordan_wigner"}
        super().__init__(
            n=n, qubittype="GridQubit", encoding_options=encoding_options, **kwargs
        )

    def _set_fock_hamiltonian(self) -> of.SymbolicOperator:
        self.fock_hamiltonian = self.fermion_operator

    def copy(self):
        self_copy = copy.deepcopy(self)
        return self_copy

    def from_json_dict(self):
        pass

    def to_json_dict(self) -> Dict:
        pass
