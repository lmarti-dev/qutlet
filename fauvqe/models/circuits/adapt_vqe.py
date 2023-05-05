import cirq
import numpy as np

from fauvqe.models.abstractmodel import AbstractModel
from fauvqe.models.fermionicModel import FermionicModel
import fauvqe.utils_cirq as cqutils
import fauvqe.utils as utils
import sympy
import openfermion as of
import itertools
import scipy
from math import prod
from typing import Union

# necessary for the full loop
from fauvqe.optimisers.optimiser import Optimiser
from fauvqe.objectives.abstractexpectationvalue import AbstractExpectationValue


def print_if_verbose(s: str, verbose: bool):
    if verbose:
        print(s)


def pauli_string_set(
    qubits: list[cirq.Qid],
    neighbour_order: int,
    pauli_list: list = None,
    periodic: bool = False,
    diagonal: bool = False,
    anti_hermitian: bool = True,
    coeff: float = 0.5,
):
    """Creates a list of all possible pauli strings on a given geometry up to some neighbour order

    Args:
        qubits (list[cirq.Qid]): the qubits on which the paulis are applied
        neighbour_order (int): the neighbour order up to which operators go. 1 means nearest-neighbour only
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
    coeff = 1 * coeff
    if anti_hermitian:
        coeff = 1j * coeff
    shape = cqutils.qubits_shape(qubits)
    numrows, numcols = shape
    for i in range(numcols * numrows):
        # get the neighbours up to the order on the grid of the given shape
        neighbours = utils.grid_neighbour_list(
            i, shape, neighbour_order, periodic=periodic, diagonal=diagonal, origin="topleft"
        )
        # do all the possible pauli strings combinations on this list of neighbours up to the given order
        max_length = len(neighbours)
        for term_order in range(1, max_length + 1):
            # all possible permutations with repetition 3**term_order
            combinations = itertools.product(pauli_list, repeat=term_order)
            for comb in combinations:
                # for each combination add a paulistring on the corresponding qubits
                dps = cirq.DensePauliString(comb, coefficient=coeff)
                pstr = dps.on(*(qubits[n] for n in neighbours[: len(comb)]))
                pauli_set.append(pstr)
    return pauli_set


def fermionic_fock_set(
    shape: tuple,
    neighbour_order: int,
    excitation_order: int,
    periodic: bool = False,
    diagonal: bool = False,
    coeff: float = 0.5,
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
        neighbours = utils.grid_neighbour_list(
            i, shape, neighbour_order, periodic=periodic, diagonal=diagonal, origin="topleft"
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
                    comb = list(utils.flatten(comb))
                    # not elegant but gotta avoid doubles
                    if (
                        i in comb
                        and comb not in all_combs
                        and list(reversed(comb)) not in all_combs
                    ):
                        term = cqutils.even_excitation(
                            coeff=coeff, indices=comb, anti_hermitian=anti_hermitian
                        )
                        fock_set.append(term)
                        all_combs.append(comb)
    return fock_set


def hamiltonian_paulisum_set(model: FermionicModel):
    """Get a set of PauliSum extracted from a FermionicModel's PauliSum hamiltonian. This method relies on an openfrmion method, so it can only be used with a FermionicModel

    Args:
        model (FermionicModel): the Fermionic model from which to extract the individual terms to create the set

    Returns:
        "list[cirq.PauliSum]": a list of PauliSums to be used in ADAPT VQE
    """
    encoded_ops = model.get_encoded_terms(anti_hermitian=True)
    paulisum_set = encoded_ops

    return paulisum_set


def fermionic_paulisum_set(model: FermionicModel, set_options: dict):
    """Get a set of PauliSum excitation operators converted using the encoding options of a certain FermionicModel and a FermionOperator set

    Args:
        model (FermionicModel): the model used to encode the FermionicOperators
        set_options (dict): options to be used with the fermionic_fock_set function
    Returns:
        "list[cirq.PauliSum]": a list of PauliSums to be used in ADAPT VQE
    """
    shape = cqutils.qubits_shape(model.flattened_qubits)
    fermionic_set = fermionic_fock_set(shape=shape, **set_options)
    paulisum_set = []

    for fop in fermionic_set:
        psum = model.encode_model(
            fermion_hamiltonian=fop,
            qubits=model.flattened_qubits,
            encoding_options=model.encoding_options,
        )
        if psum != 0:
            paulisum_set.append(psum)
    return paulisum_set


def compute_heuristic(
    ham: np.ndarray,
    op: np.ndarray,
    wf: np.ndarray,
):
    pass


def compute_gradient(
    ham: np.ndarray,
    op: np.ndarray,
    wf: np.ndarray,
    full: bool = True,
    eps: float = 1e-5,
    verbose: bool = False,
    sparse: bool = False,
) -> float:
    if sparse:
        if not full:
            # in the noiseless case dE/dt = 2Re<ps|HA|ps>
            spham = scipy.sparse.csc_matrix(ham)
            spop = scipy.sparse.csc_matrix(op)
            spwf = scipy.sparse.csc_matrix(wf)
            comm = spham @ spop
            ket = comm.dot(spwf.transpose())
            bra = spwf.conj()
            grad = float(np.abs(2 * np.real((bra @ ket).toarray())))
            print_if_verbose("gradient: {:.10f}".format(grad), verbose=verbose)
        else:
            # <p|[H,A(k)]|p> = <p|(HA(k) - A(k)H)|p> = dE/dk
            # finite diff (f(theta + eps) - f(theta - eps))/ 2eps but theta = 0
            # if A is anti hermitian
            # (<p|exp(-eps*operator) H exp(eps*operator)|p> - <p|exp(eps*operator) H exp(-eps*operator)|p>)/2eps
            # if A is hermitian the theta needs to be multiplied by 1j
            wfexp = scipy.sparse.linalg.expm_multiply(
                A=op, B=wf, start=-eps, stop=eps, num=2, endpoint=True
            )
            spham = scipy.sparse.csc_matrix(ham)
            spwf0 = scipy.sparse.csc_matrix(wfexp[0, :]).transpose()
            spwf1 = scipy.sparse.csc_matrix(wfexp[1, :]).transpose()
            grad_minus = (spwf0.getH() @ spham @ spwf0).toarray()
            grad_plus = (spwf1.getH() @ spham @ spwf1).toarray()
            grad = float(np.abs((grad_minus - grad_plus) / (2 * eps)))
    # dense methods (get it?)
    else:
        if not full:
            # dE/dt = 2Re<ps|HA|ps>
            comm = np.matmul(ham, op)
            ket = comm.dot(wf)
            bra = wf.conj()
            grad = np.abs(2 * np.real(np.dot(bra, ket)))
        else:
            # (<p|exp(-eps*operator) H exp(eps*operator)|p> - <p|exp(eps*operator) H exp(-eps*operator)|p>)/2eps
            wfexp = scipy.sparse.linalg.expm_multiply(
                A=op, B=wf, start=-eps, stop=eps, num=2, endpoint=True
            )
            wfexp_plus = wfexp[1, :]
            wfexp_minus = wfexp[0, :]
            grad_minus = wfexp_minus.conj().dot(ham @ wfexp_minus)
            grad_plus = wfexp_plus.conj().dot(ham @ wfexp_plus)
            grad = np.abs((grad_minus - grad_plus) / (2 * eps))

    if grad > 0:
        print_if_verbose("gradient: {:.10f}".format(grad), verbose=verbose)
    return grad


def filter_out_identity(psum):
    psum_out = []
    if isinstance(psum, cirq.PauliString):
        if cqutils.pauli_str_is_identity(pstr=psum):
            raise ValueError(
                "Trying to filter out the remove the identity in a PauliString consisting only of the identity: {}".format(
                    psum
                )
            )
        else:
            return psum

    for pstr in psum:
        if not cqutils.pauli_str_is_identity(pstr=pstr):
            psum_out.append(pstr)
    return cirq.PauliSum.from_pauli_strings(psum_out)


def exp_from_pauli_sum(pauli_sum: cirq.PauliSum, theta):
    psum_no_identity = filter_out_identity(pauli_sum)
    # PauliSumExponential takes cares of hermitian/anti-hermitian matters
    psum_exp = cirq.PauliSumExponential(pauli_sum_like=psum_no_identity, exponent=theta)
    return psum_exp
    # PauliSumExponential only accept A,B st exp(A)*exp(B) = exp(A+B) so might as well break them and "trotterize" them if they dont commute
    # return [cirq.PauliSumExponential(pauli_sum_like=pstr,exponent=theta) for pstr in pauli_sum if not cqutils.pauli_str_is_identity(pstr)]


def get_best_gate(
    model: AbstractModel,
    paulisum_set: list,
    param_name: str,
    tol: float,
    trial_wf: np.ndarray = None,
    initial_state: np.ndarray = None,
    verbose: bool = False,
):
    # we can set the trial wf (before computing the gradient of the energy)
    # if we want some specific psi to apply the gate to.
    # not very useful for now, might come in handy later.
    # initial_state is the state to input in the circuit
    if trial_wf is None:
        circ = cqutils.populate_empty_qubits(model=model)
        trial_wf = model.simulator.simulate(
            circ,
            param_resolver=cqutils.get_param_resolver(
                model=model, param_values=model.circuit_param_values
            ),
            initial_state=initial_state,
        ).final_state_vector

    grad_values = []
    # grad_values_full = []
    print_if_verbose("number of gates: {}".format(len(ps)), verbose=verbose)
    for ps in paulisum_set:
        ham = model.hamiltonian.matrix(qubits=model.flattened_qubits)
        op = ps.matrix(qubits=model.flattened_qubits)
        if np.any(np.conj(np.transpose(op)) != -op):
            raise ValueError("Expected op to be anti-hermitian")
        print_if_verbose("gate: {}".format(ps), verbose=verbose)
        grad_values.append(compute_gradient(ham=ham, op=op, wf=trial_wf, verbose=verbose))
    max_index = np.argmax(grad_values)
    best_ps = paulisum_set[max_index]
    if tol is not None and grad_values[max_index] < tol:
        print_if_verbose(
            "gradient ({grad:.2f}) is smaller than tol ({tol}), exiting".format(
                grad=grad_values[max_index], tol=tol
            ),
            verbose=verbose,
        )
        # iteration process is done, gradient < tol, or if stopping is set with max depth, continue
        return None
    else:
        print_if_verbose(
            "gradient ({grad}) is larger than tol ({tol}), continuing".format(
                grad=grad_values[max_index], tol=tol
            ),
            verbose=verbose,
        )
        print_if_verbose("best paulisum: {psum}".format(psum=best_ps), verbose=verbose)

        theta = sympy.Symbol("theta_{param_name}".format(param_name=param_name))
        return exp_from_pauli_sum(pauli_sum=best_ps, theta=theta), theta


def circuit_iterating_step(
    model: AbstractModel,
    paulisum_set: list,
    tol: float = 1e-15,
    trial_wf: np.ndarray = None,
    initial_state: np.ndarray = None,
    verbose: bool = False,
):
    res = get_best_gate(
        model=model,
        paulisum_set=paulisum_set,
        param_name=utils.random_word(lenw=4, Nwords=1),
        tol=tol,
        trial_wf=trial_wf,
        initial_state=initial_state,
        verbose=verbose,
    )
    if res is None:
        print_if_verbose("no best gate found, exiting", verbose=verbose)
        return True
    else:
        exp_gates, theta = res
        for gate in exp_gates:
            model.circuit += gate
        model.circuit_param.append(theta)
        cqutils.match_param_values_to_symbols(
            model=model, symbols=model.circuit_param, default_value="random"
        )
        print_if_verbose(
            "best gate found and added: {best_gate}".format(best_gate=exp_gates),
            verbose=verbose,
        )
        # print_if_verbose(model.circuit,verbose=verbose)
        return False


# pick random gate for benchmarking
def random_iterating_step(
    model: AbstractModel,
    paulisum_set: list,
    verbose: bool = False,
    default_value: str = "random",
):
    ind = np.random.choice(len(paulisum_set))
    param_name = utils.random_word(lenw=4, Nwords=1)
    theta = sympy.Symbol("theta_{param_name}".format(param_name=param_name))
    exp_gates = exp_from_pauli_sum(pauli_sum=paulisum_set[ind], theta=theta)
    for gate in exp_gates:
        model.circuit += gate
    model.circuit_param.append(theta)
    cqutils.match_param_values_to_symbols(
        model=model, symbols=model.circuit_param, default_value=default_value
    )
    print_if_verbose("adding random gate: {rgate}".format(rgate=exp_gates), verbose=verbose)
    return False


# actual entire adapt_loop
def adapt_loop(
    model: AbstractModel,
    Nsteps: int,
    paulisum_set: list[cirq.PauliSum],
    optimiser: Optimiser,
    objective: AbstractExpectationValue,
    trial_wf: np.ndarray = None,
    initial_state: np.ndarray = None,
    verbose: bool = False,
):
    result = None
    for step in range(Nsteps):
        threshold_reached = circuit_iterating_step(
            model=model,
            paulisum_set=paulisum_set,
            verbose=verbose,
            trial_wf=trial_wf,
            initial_state=initial_state,
        )
        # if the gradient is above the threshold, go on

        if threshold_reached == False:
            print("optimizing {}-th step".format(step + 1))
            # optimize to get a result everytime
            model.circuit = cqutils.populate_empty_qubits(model=model)
            result = optimiser.optimise(objective=objective)
            model.circuit_param_values = result.get_latest_step().params
            print(
                "circuit depth: {d}, number of params: {p}".format(
                    d=cqutils.depth(model.circuit), p=len(model.circuit_param)
                )
            )
        else:
            print("treshold reached, exiting")
            break

    return result, model
