from typing import Callable, Dict, Tuple, Union, Sequence, Optional
import openfermion as of
import cirq
import copy

from fauvqe.models.fermionicModel import FermionicModel
import fauvqe.utils as utils
import fauvqe.utils_cirq as cqutils


class ChemicalFermionModel(FermionicModel):
    def __init__(
        self, fermion_operator: of.FermionOperator, encoding_options: dict = None, **kwargs
    ):
        if not isinstance(fermion_operator, of.FermionOperator):
            raise TypeError("Expected a FermionOperator, got: {}".format(type(fermion_operator)))
        self.fermion_operator = fermion_operator
        n = (1, of.count_qubits(operator=self.fermion_operator))
        if encoding_options is None:
            encoding_options = {"encoding_name": "jordan_wigner"}
        super().__init__(n=n, qubittype="GridQubit", encoding_options=encoding_options, **kwargs)

    def _set_fock_hamiltonian(self) -> of.SymbolicOperator:
        self.fock_hamiltonian = self.fermion_operator

    def _get_initial_state(
        self, name: str, initial_state: Union[int, Sequence[int]], Nf
    ) -> cirq.OP_TREE:
        self.Nf = Nf
        if name == "none":
            return []
        elif name == "computational" or name == "hadamard":
            if initial_state is None:
                raise ValueError("initial_state cannot be None")
            if isinstance(initial_state, int):
                # convert int to bin and then index
                initial_state = utils.index_bits(bin(initial_state))
            op_tree = [cirq.X(self.flattened_qubits[ind]) for ind in initial_state]
            if name == "hadamard":
                op_tree.extend([cirq.H(q) for q in self.flattened_qubits])
            return op_tree
        else:
            raise NameError("No initial state named {}".format(name))

    def copy(self):
        self_copy = copy.deepcopy(self)
        return self_copy

    def from_json_dict(self):
        pass

    def to_json_dict(self) -> Dict:
        pass
