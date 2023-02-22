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
    qubits: "list[cirq.Qid]",
    min_len: int,
    max_len: int,
    pauli_list: list = None,
    anti_hermitian: bool = True,
):
    pauli_set = []
    if pauli_list is None:
        pauli_list = ["X", "Y", "Z"]
    coefficient = 1
    if anti_hermitian:
        coefficient = 1j
    for l in range(min_len, max_len + 1):
        combinations = itertools.combinations_with_replacement(pauli_list, l)
        for comb in combinations:
            dps = cirq.DensePauliString(comb, coefficient=coefficient)
            for start in range(len(qubits)):
                i1 = start
                i2 = start + len(comb)
                if i2 <= len(qubits):
                    pstr = dps.on(*qubits[i1:i2])
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
        shape (Tuple of int): Shape of the grid, ie the number of rows and columns. For now, this only works wth 2D square constructs
        neighbour_order (int): Highest (nth) nearest neighbour order of hopping terms.
        excitation_order (int): How many fock operators are in the terms. First order is ij, 2nd order is ijkl, third is ijklmn and so on.
        periodic (bool, optional): Whether the borders of the grid have periodic boundary conditions. Defaults to False.
        diagonal: (bool): Whether the operators should wrap around the border.Defaults to False
        coeff: (float): The coefficient in front of each operator. Defaults to 0.5,
        anti_hermitian: (bool) Whether to ensure that the operators are anti hermitiant, so that they can be exponentiated into unitaries without being multiplied by 1j. Defaults to True
    """
    numrows, numcols = shape
    fock_set = []
    for i in range(numcols * numrows):
        neighbours = utils.grid_neighbour_list(
            i, shape, neighbour_order, periodic=periodic, diagonal=diagonal
        )
        for term_order in range(2, 2 ** (excitation_order) + 1, 2):
            combinations = itertools.combinations_with_replacement(
                neighbours[1:], term_order - 1
            )
            for comb in combinations:
                fock_set.append(
                    cqutils.even_excitation(
                        coeff=coeff, indices=(i,) + comb, anti_hermitian=anti_hermitian
                    )
                )
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


def compute_gradient(
    ham: np.ndarray,
    op: np.ndarray,
    wf: np.ndarray,
    full: bool = False,
    eps: float = 1e-5,
    verbose: bool = False,
) -> float:
    if not full:
        # in the noiseless case dE/dt = 2Re<ps|HA|ps>
        spham = scipy.sparse.csc_matrix(ham)
        spop = scipy.sparse.csc_matrix(op)
        comm = spham @ spop
        ket = comm.dot(wf)
        bra = wf.conj()
        grad = np.abs(2 * np.real(np.dot(bra, ket)))
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
        grad_minus = wfexp[0, :].conj().dot(ham @ wfexp[0, :])
        grad_plus = wfexp[1, :].conj().dot(ham @ wfexp[1, :])
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

    for ps in paulisum_set:
        ham = model.hamiltonian.matrix(qubits=model.flattened_qubits)
        op = ps.matrix(qubits=model.flattened_qubits)
        if np.any(utils.unitary_transpose(op) != -op):
            raise ValueError("Expected op to be anti-hermitian")
        print_if_verbose("gate: {}".format(ps), verbose=verbose)
        grad_values.append(
            compute_gradient(ham=ham, op=op, wf=trial_wf, verbose=verbose)
        )
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
    default_value="random",
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
    print_if_verbose(
        "adding random gate: {rgate}".format(rgate=exp_gates), verbose=verbose
    )
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
        else:
            print("treshold reached, exiting")
            break

    return result, model
