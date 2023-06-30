from typing import Callable, Dict, Tuple, Union, Sequence, Optional
import openfermion as of
import cirq
import copy

from fauvqe.models.fermionicModel import FermionicModel


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

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "n": self.n,
                "fermion_operator": self.fermion_operator,
                "encoding_options": self.encoding_options,
            },
            "params": {
                "circuit": self.circuit,
                "circuit_param": self.circuit_param,
                "circuit_param_values": self.circuit_param_values,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        inst = cls(**dct["constructor_params"])

        inst.circuit = dct["params"]["circuit"]
        inst.circuit_param = dct["params"]["circuit_param"]
        inst.circuit_param_values = dct["params"]["circuit_param_values"]

        return inst
