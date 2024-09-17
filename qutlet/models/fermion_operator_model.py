from typing import Union
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

        if qubit_shape is None:
            qubit_shape = fermion_op_sites_number(fermion_operator)
        super().__init__(
            qubit_shape=qubit_shape, encoding_options=encoding_options, **kwargs
        )

    def _set_fock_hamiltonian(self) -> of.SymbolicOperator:
        self.fock_hamiltonian = self.fermion_operator

    @property
    def __to_json__(self) -> dict:
        return {
            "fermion_operator": self.fermion_operator,
            "encoding_options": self.encoding_options,
        }

    @property
    def non_interacting_model(self) -> "FermionOperatorModel":
        return quadratic_model(self)


def quadratic_model(model: FermionicModel) -> FermionOperatorModel:
    return FermionOperatorModel(
        of.get_fermion_operator(model.quadratic_terms),
        n_electrons=model.n_electrons,
        qubit_shape=model.qubit_shape,
    )


def non_quadratic_model(model: FermionicModel) -> FermionOperatorModel:
    return FermionOperatorModel(
        of.get_fermion_operator(model.non_quadratic_terms),
        n_electrons=model.n_electrons,
        qubit_shape=model.qubit_shape,
    )
