import itertools

import cirq
import numpy as np
import sympy

from qutlet.models import FermionicModel
from qutlet.utilities import (
    all_pauli_str_commute,
    even_excitation,
    flatten,
    grid_neighbour_list,
    qubits_shape,
    pauli_str_is_identity,
    jw_get_free_couplers,
)
import abc
from typing import Union
import re


def filter_out_identity(psum):
    psum_out = []
    if isinstance(psum, cirq.PauliString):
        if pauli_str_is_identity(pstr=psum):
            raise ValueError(
                "Trying to filter out the remove the identity in a PauliString consisting only of the identity: {}".format(
                    psum
                )
            )
        else:
            return psum

    for pstr in psum:
        if not pauli_str_is_identity(pstr=pstr):
            psum_out.append(pstr)
    return cirq.PauliSum.from_pauli_strings(psum_out)


def Sdagger(qubit):
    return cirq.MatrixGate(
        matrix=np.array([[1, 0], [0, -1j]]),
        qid_shape=(2,),
        unitary_check=False,
        name="S†",
    ).on(qubit)


def pauli_string_exponentiating_circuit(
    pauli_string: cirq.PauliString, theta: sympy.Symbol, right_to_left: bool
):
    # check that we can exponentiate the pauli string
    coeff = pauli_string.coefficient
    atol = 1e-8
    if abs(coeff.imag) > atol and abs(coeff.real) > atol:
        raise ValueError(
            "Pauli string is neither fully Hermitian nor anti Hermitian. Full coeff {}".format(
                coeff
            )
        )
    coeff_multiplier = -1j if abs(coeff.imag) > atol else 1.0
    # init stuff
    op_tree = []
    qubits = [qubit for qubit in pauli_string]
    # sort by qubit number
    indices = np.argsort(qubits)
    qubits = [qubits[idx] for idx in indices]
    paulis = [pauli_string.gate[int(idx)] for idx in indices]
    first_layer = []
    last_layer = []
    if right_to_left:
        cnot_layer = [
            cirq.CNOT(qubits[q], qubits[q - 1]) for q in range(len(qubits) - 1, 0, -1)
        ]
    else:
        cnot_layer = [
            cirq.CNOT(qubits[q], qubits[q + 1]) for q in range(len(qubits) - 1)
        ]
    for qubit, pauli in zip(qubits, paulis):
        # print(qubit, pauli)
        if pauli == cirq.X:
            first_layer += (cirq.H(qubit),)
            last_layer += (cirq.H(qubit),)
        elif pauli == cirq.Y:
            # first_layer += (cirq.Rx(rads=-1).on(qubit),)
            # last_layer += (cirq.Rx(rads=1).on(qubit),)
            # this doens't work
            first_layer += (Sdagger(qubit), cirq.H(qubit))
            last_layer += (cirq.H(qubit), cirq.S(qubit))
        elif pauli == cirq.Z or pauli == cirq.I:
            pass
            # first_layer += (cirq.I(qubit),)
            # last_layer += (cirq.I(qubit),)
        else:
            raise ValueError("the operation is not a Pauli matrix")
    op_tree += first_layer
    op_tree += cnot_layer
    # same as paulisumexponential

    if right_to_left:
        rz_qubit = 0
    else:
        rz_qubit = -1
    op_tree += (
        cirq.Rz(rads=-2 * coeff_multiplier * coeff * theta / np.pi).on(
            qubits[rz_qubit]
        ),
    )
    op_tree += cnot_layer[::-1]
    op_tree += last_layer
    return op_tree


def pauli_sum_exponentiating_circuit(
    pauli_sum: cirq.PauliSum, theta: sympy.Symbol, right_to_left: bool
):
    op_tree = []
    for pstr in pauli_sum:
        op_tree += pauli_string_exponentiating_circuit(
            pauli_string=pstr, theta=theta, right_to_left=right_to_left
        )
    return op_tree


def exp_from_pauli_sum(pauli_sum: cirq.PauliSum, theta, use_circuit=False):
    psum_no_identity = filter_out_identity(pauli_sum)
    if use_circuit:
        return pauli_sum_exponentiating_circuit(
            pauli_sum=psum_no_identity, theta=theta, right_to_left=False
        )
    else:
        # PauliSumExponential takes cares of hermitian/anti-hermitian matters
        return cirq.PauliSumExponential(pauli_sum_like=psum_no_identity, exponent=theta)
        # PauliSumExponential only accept A,B st exp(A)*exp(B) = exp(A+B) so might as well break them and "trotterize" them if they dont commute
        # return [cirq.PauliSumExponential(pauli_sum_like=pstr,exponent=theta) for pstr in pauli_sum if not pauli_str_is_identity(pstr)]


def get_psum_str_pstr(psum_str: str):
    # get all pstr elements from the psum_str without plusses and minuses
    return re.split("[+-]", re.sub("\s", "", psum_str))


def split_psum_str(psum_str: str) -> list:
    """Splits a Pauli sum into a list of Pauli Strings

    Args:
        psum_str (str): a string representing a pauli sum

    Returns:
        list: the list of Pauli Strings
    """
    return re.split("([+-])", re.sub("\s", "", psum_str))


def len_psum_str(psum_str: str):
    pstrs = split_psum_str(psum_str=psum_str)
    return max([len(s) for s in pstrs])


def psum_str_qubit_req(psum_str: str):
    pstrs = split_psum_str(psum_str=psum_str)
    return sum([len(item) for item in pstrs if "+" not in item and "-" not in item])


def process_psum_str(psum_str: str, qubits: list[cirq.Qid], coeff: float = 1):
    # turns a psum string XX + YY into a cirq psum with qubtis
    pstrs = split_psum_str(psum_str=psum_str)
    psum = cirq.PauliSum()
    parity = 1
    qubit_count = 0
    for item in pstrs:
        if "-" in item:
            parity = -1
        elif "+" in item:
            parity = 1
        else:
            dps = cirq.DensePauliString(item, coefficient=coeff)
            pstr = dps.on(*qubits[qubit_count : qubit_count + len(dps)])
            psum += parity * pstr
            qubit_count += len(dps)
    return psum


class GatePool:
    def __init__(self) -> None:
        pass

    def _set_operator_pool(self, pool: list):
        self.operator_pool = pool
        self.verify_gate_pool()

    @abc.abstractmethod
    def gate_from_op(self, ind) -> Union[cirq.CIRCUIT_LIKE, sympy.Symbol]:
        raise NotImplementedError

    def qubits(self, ind):
        op = self.operator_pool[ind]
        if isinstance(op, cirq.PauliSum) or isinstance(op, cirq.PauliString):
            return op.qubits
        elif isinstance(op, cirq.Circuit):
            return op.all_qubits()

    @abc.abstractmethod
    def verify_gate_pool(self):
        raise NotImplementedError

    @abc.abstractmethod
    def matrix(self, ind, qubits):
        raise NotImplementedError


class ExponentiableGatePool(GatePool):
    def gate_from_op(self, ind, param_name, opts={}):
        pauli_sum = self.operator_pool[ind]
        theta = sympy.Symbol("theta_{param_name}".format(param_name=param_name))
        return exp_from_pauli_sum(pauli_sum=pauli_sum, theta=theta, **opts), theta

    def verify_gate_pool(self):
        print("Verifying gate pool, {} gates".format(len(self.operator_pool)))
        for ind in range(len(self.operator_pool)):
            opmat = self.matrix(ind, qubits=self.qubits(ind))
            if np.any(np.conj(np.transpose(opmat)) != -opmat):
                raise ValueError("Expected op to be anti-hermitian")
        print("All gates are anti-hermitian, proceeding.")

    def matrix(self, ind, qubits):
        return self.operator_pool[ind].matrix(qubits=qubits)


class PauliSumListSet(ExponentiableGatePool):
    """Provide a list of pauli sums in a string format, and get all possible exponentiable gates"""

    def __init__(
        self,
        qubits: list[cirq.Qid],
        neighbour_order: int,
        psum_list: list[str],
        k_locality: int,
        periodic: bool = False,
        diagonal: bool = True,
        anti_hermitian: bool = True,
        coeff: float = 1,
    ):
        pauli_list_set = []
        if anti_hermitian:
            coeff = 1j * np.abs(coeff)
        shape = qubits_shape(qubits)
        numrows, numcols = shape
        for psum_str in psum_list:
            for i in range(numcols * numrows):
                # get the neighbours up to the order on the grid of the given shape
                # do all the possible pauli strings combinations on this list of neighbours up to the given order
                neighbours = grid_neighbour_list(
                    i,
                    shape,
                    neighbour_order,
                    periodic=periodic,
                    diagonal=diagonal,
                    origin="topleft",
                )
                # get the correct qubit combinations, without having the same qubit on the same pstr,
                # such that you don't get spurrious multiplications, i.e X0X0 = I and
                # adapt breaks.
                # so you get all combs w/o reps for each pstr in the psum
                # and then you product them quintuple for loop style
                combs = itertools.product(
                    *(
                        itertools.combinations(range(len(neighbours)), len(pstr))
                        for pstr in get_psum_str_pstr(psum_str)
                    )
                )
                for comb in set(combs):
                    if k_locality == "auto":
                        n_qubits_req = len_psum_str(psum_str=psum_str)
                    elif isinstance(k_locality, int):
                        n_qubits_req = k_locality
                    else:
                        raise ValueError(
                            'expected k-locality to be int or "auto", got {}'.format(
                                k_locality
                            )
                        )
                    if len(list(set(flatten(comb)))) <= n_qubits_req:
                        pauli_sum_qubits = [qubits[c] for c in flatten(comb)]
                        processed_pauli_sum = process_psum_str(
                            psum_str=psum_str, qubits=pauli_sum_qubits, coeff=coeff
                        )
                        # if you pauli sum has cross terms XY-YX you might get a null term..
                        # also remove the identity operators
                        if not all(
                            pauli_str_is_identity(pstr) for pstr in processed_pauli_sum
                        ) and processed_pauli_sum != cirq.PauliSum(cirq.LinearDict({})):
                            pauli_list_set.append(processed_pauli_sum)
        self._set_operator_pool(pauli_list_set)


class PauliStringSet(ExponentiableGatePool):
    def __init__(
        self,
        qubits: list[cirq.Qid],
        neighbour_order: int,
        max_length: int = None,
        pauli_list: list = None,
        periodic: bool = False,
        diagonal: bool = False,
        anti_hermitian: bool = True,
        coeff: float = 1,
    ) -> None:
        """Creates a list of all possible pauli strings on a given geometry up to some neighbour order

        Args:
            qubits (list[cirq.Qid]): the qubits on which the paulis are applied
            neighbour_order (int): the neighbour order up to which operators go. 1 means nearest-neighbour only
            max_length (int): the max length of Pauli strings. If None, defaults to length of neighbour list.
            pauli_list (list, optional): The list of available Pauli matrices. Defaults to None which means all 3 are used.
            periodic (bool, optional): whether the bounds of the qubit lattice are periodic. Defaults to False.
            diagonal (bool, optional): Whether to consider diagonal neighbours. Defaults to False.
            anti_hermitian (bool, optional): whether to make the Pauli string anti-hermitian. Defaults to True.
            coeff (float, optional): the default coefficient of the pauli matrix. Defaults to 0.5.

        Returns:
            list[cirq.PauliString]: List of pauli strings
        """
        pauli_set = []
        if pauli_list is None:
            # add I in case we want to go with non-local neighbours
            pauli_list = ["X", "Y", "Z"]
        if anti_hermitian:
            coeff = 1j * np.abs(coeff)
        shape = qubits_shape(qubits)
        numrows, numcols = shape
        for i in range(numcols * numrows):
            # get the neighbours up to the order on the grid of the given shape
            neighbours = grid_neighbour_list(
                i,
                shape,
                neighbour_order,
                periodic=periodic,
                diagonal=diagonal,
                origin="topleft",
            )
            # do all the possible pauli strings combinations on this list of neighbours up to the given order
            if max_length is None:
                max_length = len(neighbours)
            for term_order in range(1, min(max_length + 1, len(neighbours))):
                # all possible permutations with repetition 3**term_order
                combinations = itertools.product(pauli_list, repeat=term_order)
                for comb in combinations:
                    # for each combination add a paulistring on the corresponding qubits
                    dps = cirq.DensePauliString(comb, coefficient=coeff)
                    pstr = dps.on(*(qubits[n] for n in neighbours[: len(comb)]))
                    pauli_set.append(pstr)
        self._set_operator_pool(pauli_set)


class PauliSumSet(PauliStringSet):
    def __init__(
        self,
        qubits: list[cirq.Qid],
        neighbour_order: int,
        max_length: int = None,
        pauli_list: list = None,
        periodic: bool = False,
        diagonal: bool = False,
        anti_hermitian: bool = True,
        coeff: float = 1,
        sum_order: int = 2,
    ):
        super.__init__(
            qubits=qubits,
            neighbour_order=neighbour_order,
            max_length=max_length,
            pauli_list=pauli_list,
            periodic=periodic,
            diagonal=diagonal,
            anti_hermitian=anti_hermitian,
            coeff=coeff,
        )
        non_commuting_pauli_sums_set = []
        combs = itertools.combinations(self.pool, sum_order)
        for comb in combs:
            psum = sum(comb)
            if all_pauli_str_commute(psum):
                non_commuting_pauli_sums_set.append(psum)
        self._set_operator_pool(non_commuting_pauli_sums_set)


def get_pauli_sums_from_hamiltonian(model: FermionicModel, threshold: float = None):
    encoded_ops = model.get_encoded_terms(anti_hermitian=True)
    if threshold is not None:
        remaining_encoded_ops = []
        for psum in encoded_ops:
            psum_list = [term for term in psum]
            if len(psum_list):
                mean_coeff = np.mean([np.abs(term.coefficient) for term in psum_list])
                if mean_coeff > threshold:
                    remaining_encoded_ops.append(psum)
        pauli_sum_set = remaining_encoded_ops
    else:
        pauli_sum_set = encoded_ops
    return pauli_sum_set


class HamiltonianPauliSumSet(ExponentiableGatePool):
    def __init__(self, model: FermionicModel, threshold: float = None):
        """Get a set of PauliSum extracted from a FermionicModel's PauliSum hamiltonian. This method relies on an openfrmion method, so it can only be used with a FermionicModel

        Args:
            model (FermionicModel): the Fermionic model from which to extract the individual terms to create the set

        Returns:
            "list[cirq.PauliSum]": a list of PauliSums to be used in ADAPT VQE
        """
        pauli_sum_set = get_pauli_sums_from_hamiltonian(model, threshold)

        self._set_operator_pool(pauli_sum_set)

        return pauli_sum_set


def fermionic_fock_set(
    shape: tuple,
    neighbour_order: int,
    excitation_order: int,
    periodic: bool = False,
    diagonal: bool = False,
    coeff: float = 1,
    anti_hermitian: bool = True,
):
    """Creates a list of Hermitian fock terms for a grid of fermions of the given shape

    Args:
        shape (Tuple of int): Shape of the fermionic grid, with one fermionic DOF per site, ie the number of rows and columns. For now, this only works wth 2D square constructs
        neighbour_order (int): Highest (nth) nearest neighbour order of hopping terms.
        excitation_order (int): How many fock operators are in the terms. 1st order is ij, 2nd order is ijkl, 3rd is ijklmn and so on.
        periodic (bool, optional): Whether the borders of the grid have periodic boundary conditions. Defaults to False.
        diagonal: (bool): Whether the operators should wrap around the border.Defaults to False
        coeff: (float): The coefficient in front of each operator. Defaults to 0.5,
        anti_hermitian: (bool) Whether to ensure that the operators are anti hermitiant, so that they can be exponentiated into unitaries without being multiplied by 1j. Defaults to True
    """
    all_combs = []
    numrows, numcols = shape
    fock_set = []
    for i in range(numcols * numrows):
        neighbours = grid_neighbour_list(
            i,
            shape,
            neighbour_order,
            periodic=periodic,
            diagonal=diagonal,
            origin="topleft",
        )
        for term_order in range(2, 2 * (excitation_order) + 1, 2):
            half_order = int(term_order / 2)
            # if there are enough neighbours to get a term go on
            if len(neighbours) >= half_order:
                # get all combinations of non repeating indices for each half of the fock term
                # no repetitions because two same fock terms put the thing to zero
                half_combs = list(itertools.combinations(neighbours, half_order))
                # products of all possible fock term halves
                combinations = itertools.product(half_combs, half_combs)
                for comb in combinations:
                    # flatten
                    comb = list(flatten(comb))
                    # not elegant but gotta avoid doubles
                    if (
                        i in comb
                        and comb not in all_combs
                        and list(reversed(comb)) not in all_combs
                    ):
                        term = even_excitation(
                            coeff=coeff, indices=comb, anti_hermitian=anti_hermitian
                        )
                        fock_set.append(term)
                        all_combs.append(comb)
    return fock_set


class FreeCouplersSet(ExponentiableGatePool):
    def __init__(self, model: FermionicModel, set_options: dict):
        if "zero_index" not in set_options:
            set_options["zero_index"] = 0
        free_couplers = jw_get_free_couplers(model, **set_options, hc=True)
        free_couplers = [1j * coupler for coupler in free_couplers]
        self._set_operator_pool(free_couplers)


class FermionicPauliSumSet(ExponentiableGatePool):
    def __init__(self, model: FermionicModel, set_options: dict):
        """Get a set of PauliSum excitation operators converted using the encoding options of a certain FermionicModel and a FermionOperator set

        Args:
            model (FermionicModel): the model used to encode the FermionicOperators
            set_options (dict): options to be used with the fermionic_fock_set function
        Returns:
            "list[cirq.PauliSum]": a list of PauliSums to be used in ADAPT VQE
        """
        shape = qubits_shape(model.qubits)
        fermionic_set = fermionic_fock_set(shape=shape, **set_options)
        paulisum_set = []

        for fop in fermionic_set:
            psum = model.encode_fermion_operator(
                fermion_hamiltonian=fop,
                qubits=model.qubits,
                encoding_options=model.encoding_options,
            )
            if psum != 0:
                paulisum_set.append(psum)
        self._set_operator_pool(paulisum_set)


class SubCircuitSet(GatePool):
    """Get a set of sub circuits as legos for your vqe, you merely need to create a circuit with dummy qubits, and pass it to the init"""

    def __init__(
        self,
        qubits: list[cirq.Qid],
        neighbour_order: int,
        sub_circ: cirq.CIRCUIT_LIKE,
        symmetric: bool = False,
    ):
        circ = cirq.Circuit(sub_circ)
        k_locality = len(circ.all_qubits())

        # if your gate can be flipped wrt qubits (eg FSIM), then use combinations otherwise (eg Controlled-Ry) use perms
        if symmetric:
            combs = itertools.combinations(range(len(qubits)), r=k_locality)
        else:
            combs = itertools.permutations(range(len(qubits)), r=k_locality)
        gate_pool = []
        for comb in combs:
            if abs(max(comb) - min(comb)) <= neighbour_order:
                used_qubits = (qubits[x] for x in comb)
                transformed_circ = circ.transform_qubits(
                    {k: v for k, v in zip(sorted(circ.all_qubits()), used_qubits)}
                )

                gate_pool.append(transformed_circ)
        self._set_operator_pool(gate_pool)

    def gate_from_op(self, ind, param_name):
        circ = self.operator_pool[ind]
        symbols = cirq.parameter_symbols(circ)
        param_dict = {
            sym: sympy.Symbol(param_name + "_{}".format(ind))
            for ind, sym in enumerate(symbols)
        }
        param_res = cirq.ParamResolver(param_dict)
        resolved_circ = cirq.resolve_parameters(circ, param_resolver=param_res)

        return resolved_circ, tuple(param_dict.values())

    def verify_gate_pool(self):
        # hmm perhaps something might come here
        pass
