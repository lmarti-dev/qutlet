from typing import Dict, Union
import openfermion as of
from qutlet.utilities import fermion_op_sites_number

from qutlet.models.fermionic_model import FermionicModel


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
        if qubit_shape is None:
            qubit_shape = fermion_op_sites_number(fermion_operator)
        super().__init__(
            qubit_shape=qubit_shape, encoding_options=encoding_options, **kwargs
        )

    def _set_fock_hamiltonian(self) -> of.SymbolicOperator:
        self.fock_hamiltonian = self.fermion_operator

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


def quadratic_model(model: FermionicModel):
    return FermionOperatorModel(model.quadratic_terms)


def non_quadratic_model(model: FermionicModel):
    return FermionOperatorModel(model.non_quadratic_terms)
