from typing import Callable, Dict, Tuple, Union, Sequence, Optional
import openfermion as of
import cirq
import copy

from qutlet.models.fermionicModel import FermionicModel


class FermionOperatorModel(FermionicModel):
    def __init__(
        self,
        fermion_operator: of.FermionOperator,
        encoding_options: dict = None,
        qubit_shape: Union[tuple, int] = None,
        **kwargs
    ):
        if not isinstance(fermion_operator, of.FermionOperator):
            raise TypeError(
                "Expected a FermionOperator, got: {}".format(type(fermion_operator))
            )
        self.fermion_operator = fermion_operator
        if encoding_options is None:
            encoding_options = {"encoding_name": "jordan_wigner"}
        super().__init__(
            qubit_shape=qubit_shape, encoding_options=encoding_options, **kwargs
        )

    def _set_fock_hamiltonian(self) -> of.SymbolicOperator:
        self.fock_hamiltonian = self.fermion_operator

    def copy(self):
        self_copy = copy.deepcopy(self)
        return self_copy

    def __to_json__(self) -> Dict:
        return {
            "constructor_params": {
                "fermion_operator": self.fermion_operator,
                "encoding_options": self.encoding_options,
            },
        }

    @classmethod
    def from_dict(cls, dct: Dict):
        inst = cls(**dct["constructor_params"])

        return inst
