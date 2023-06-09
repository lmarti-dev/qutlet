import cirq
import numpy as np

from fauvqe.models.abstractmodel import AbstractModel
from fauvqe.models.fermionicModel import FermionicModel
import sympy
import openfermion as of
import itertools
import scipy
from math import prod
from typing import Tuple, Union

# necessary for the full loop
from fauvqe.optimisers.optimiser import Optimiser
from fauvqe.optimisers.optimisation_result import OptimisationResult
from fauvqe.objectives.abstractexpectationvalue import AbstractExpectationValue
import fauvqe.utilities.circuit
import fauvqe.utilities.fermion
import fauvqe.utilities.generic

import multiprocessing as mp
from fauvqe.models.abstractmodel import AbstractModel


class ADAPT:
    def __init__(
        self,
        model: AbstractModel,
        gate_pool_options: Union[dict, list],
        measure: dict,
        preprocess: bool,
        verbose: bool,
    ):
        self.model = model
        self.measure = measure
        self.verbose = verbose
        self.preprocess = preprocess
        if isinstance(gate_pool_options, list):
            self.pool_name = "custom"
            self.gate_pool = gate_pool_options
        elif isinstance(gate_pool_options, dict):
            self.pool_name = gate_pool_options.pop("pool_name", None)
            self.gate_pool = self.pick_gate_pool(
                pool_name=self.pool_name, gate_pool_options=gate_pool_options
            )

        self.verify_gate_pool(self.gate_pool)
        if self.preprocess:
            self.preprocessed_ops = self.preprocess_gate_gradients()

    def verify_gate_pool(self, gate_pool):
        self.verbose_print("Verifying gate pool, {} gates".format(len(gate_pool)))
        for op in gate_pool:
            opmat = op.matrix(qubits=self.model.flattened_qubits)
            if np.any(np.conj(np.transpose(opmat)) != -opmat):
                raise ValueError("Expected op to be anti-hermitian")

    def measure_gradient(self, ind: int, trial_state: np.ndarray):
        self.verbose_print(
            "computing {g} gradient, preproc: {p}".format(
                g=self.gate_pool[ind], p=self.preprocess
            )
        )
        if self.measure["name"] == "energy":
            if self.preprocess:
                grad = compute_preprocessed_energy_gradient(
                    trial_state=trial_state,
                    preprocessed_energy_op=self.preprocessed_ops[ind],
                )
            else:
                grad = compute_energy_gradient(
                    ham=self.model.hamiltonian.matrix(
                        qubits=self.model.flattened_qubits
                    ),
                    op=self.gate_pool[ind].matrix(qubits=self.model.flattened_qubits),
                    trial_state=trial_state,
                    full=True,
                    eps=1e-5,
                    sparse=False,
                )
        if self.measure["name"] == "fidelity":
            if self.preprocess:
                grad = compute_preprocessed_fidelity_gradient(
                    trial_state=trial_state,
                    preprocessed_fid_state=self.preprocessed_ops[ind],
                )
            else:
                grad = compute_fidelity_gradient(
                    op=self.gate_pool[ind].matrix(qubits=self.model.flattened_qubits),
                    trial_state=trial_state,
                    target_state=self.measure["target_state"],
                )
        self.verbose_print("gradient: {:.10f}".format(grad))
        return grad

    def verbose_print(self, s: str):
        if self.verbose:
            print(s)

    def pick_gate_pool(self, pool_name, gate_pool_options):
        if pool_name == "pauli_string":
            set_options = {"qubits": self.model.flattened_qubits}
            set_options.update(gate_pool_options)
            return pauli_string_set(**set_options)
        elif pool_name == "hamiltonian":
            return hamiltonian_paulisum_set(model=self.model)
        elif pool_name == "fermionic":
            return fermionic_paulisum_set(
                model=self.model, set_options=gate_pool_options
            )

    def preprocess_gate_gradients(self):
        self.verbose_print(
            "Preprocessing {n} gates of {p} pool for {m}".format(
                n=len(self.gate_pool), p=self.pool_name, m=self.measure["name"]
            )
        )
        preprocess_opts = {"ham": self.model.hamiltonian}
        preprocess_ops = []
        if self.measure["name"] == "energy":
            preprocess_fn = preprocess_for_energy
        elif self.measure["name"] == "fidelity":
            preprocess_fn = preprocess_for_fidelity
            preprocess_opts.update(
                {
                    "target_state": self.measure["target_state"],
                }
            )
        for op in self.gate_pool:
            preprocess_ops.append(preprocess_fn(op=op, **preprocess_opts))
        self.verbose_print("Preprocessing over")
        return preprocess_ops

    def get_best_gate(
        self,
        tol: float,
        trial_state: np.ndarray = None,
        initial_state: np.ndarray = None,
        n_jobs: int = 1,
    ):
        # we can set the trial wf (before computing the gradient of the energy)
        # if we want some specific psi to apply the gate to.
        # not very useful for now, might come in handy later.
        # initial_state is the state to input in the circuit
        if trial_state is None:
            circ = fauvqe.utilities.circuit.populate_empty_qubits(model=self.model)
            trial_state = self.model.simulator.simulate(
                circ,
                param_resolver=fauvqe.utilities.circuit.get_param_resolver(
                    model=self.model, param_values=self.model.circuit_param_values
                ),
                initial_state=initial_state,
            ).final_state_vector

        grad_values = []
        self.verbose_print("Gate pool size: {}".format(len(self.gate_pool)))
        self.verbose_print(
            "measure chosen: {}, preprocessing: {}".format(
                self.measure["name"], self.preprocess
            )
        )
        # TODO: implement multiprocessing
        if n_jobs > 1:
            pass
        for ind in range(len(self.gate_pool)):
            grad = self.measure_gradient(ind=ind, trial_state=trial_state)
            grad_values.append(grad)
        max_index = np.argmax(grad_values)
        best_ps = self.gate_pool[max_index]
        if tol is not None and grad_values[max_index] < tol:
            self.verbose_print(
                "gradient ({grad:.2f}) is smaller than tol ({tol}), exiting".format(
                    grad=grad_values[max_index], tol=tol
                )
            )
            # iteration process is done, gradient < tol, or if stopping is set with max depth, continue
            return None
        else:
            self.verbose_print(
                "gradient ({grad}) is larger than tolerance: ({tol}), continuing".format(
                    grad=grad_values[max_index], tol=tol
                ),
            )
            self.verbose_print("best Pauli sum: {psum}".format(psum=best_ps))

            param_name = param_name_from_depth(circ=self.model.circuit)
            theta = sympy.Symbol("theta_{param_name}".format(param_name=param_name))
            return exp_from_pauli_sum(pauli_sum=best_ps, theta=theta), theta

    def append_best_gate_to_circuit(
        self,
        tol: float = 1e-15,
        trial_state: np.ndarray = None,
        initial_state: np.ndarray = None,
    ) -> bool:
        res = self.get_best_gate(
            tol=tol,
            trial_state=trial_state,
            initial_state=initial_state,
        )
        if res is None:
            self.verbose_print("No best gate found or threshold reached, exiting")
            return True
        else:
            exp_gates, theta = res
            for gate in exp_gates:
                self.model.circuit += gate
            self.model.circuit_param.append(theta)
            fauvqe.utilities.circuit.match_param_values_to_symbols(
                model=self.model,
                symbols=self.model.circuit_param,
                default_value="random",
            )
            self.verbose_print(
                "best gate found and added: {best_gate}".format(best_gate=exp_gates)
            )
            return False

    # pick random gate for benchmarking
    def append_random_gate_to_circuit(
        self,
        default_value: str = "random",
    ):
        ind = np.random.choice(len(self.gate_pool))
        param_name = param_name_from_depth(circ=self.model.circuit)
        theta = sympy.Symbol("theta_{param_name}".format(param_name=param_name))
        exp_gates = exp_from_pauli_sum(pauli_sum=self.gate_pool[ind], theta=theta)
        for gate in exp_gates:
            self.model.circuit += gate
        self.model.circuit_param.append(theta)
        fauvqe.utilities.circuit.match_param_values_to_symbols(
            model=self.model,
            symbols=self.model.circuit_param,
            default_value=default_value,
        )
        self.verbose_print("adding random gate: {rgate}".format(rgate=exp_gates))
        return False

    def loop(
        self,
        n_steps: int,
        optimiser: Optimiser,
        objective: AbstractExpectationValue,
        trial_state: np.ndarray = None,
        initial_state: np.ndarray = None,
        callback: callable = None,
    ) -> Tuple[OptimisationResult, AbstractModel]:
        result = None
        for step in range(n_steps):
            threshold_reached = self.append_best_gate_to_circuit(
                trial_state=trial_state, initial_state=initial_state
            )
            # if the gradient is above the threshold, go on
            if threshold_reached == False:
                self.verbose_print("optimizing {}-th step".format(step + 1))
                # optimize to get a result everytime
                self.model.circuit = fauvqe.utilities.circuit.populate_empty_qubits(
                    model=self.model
                )
                result = optimiser.optimise(objective=objective)
                self.model.circuit_param_values = result.get_latest_step().params
                print(
                    "circuit depth: {d}, number of params: {p}".format(
                        d=fauvqe.utilities.circuit.depth(self.model.circuit),
                        p=len(self.model.circuit_param),
                    )
                )
                if callback is not None:
                    callback(result)
            else:
                print("treshold reached, exiting")
                break

        return result

    def pool_gate_measure_jobs(n_jobs, data):
        pool = mp.Pool(n_jobs)
        result = pool.map(x, data)


def param_name_from_depth(circ: cirq.Circuit) -> str:
    return "p_" + str(fauvqe.utilities.circuit.depth(circ))


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
    shape = fauvqe.utilities.circuit.qubits_shape(qubits)
    numrows, numcols = shape
    for i in range(numcols * numrows):
        # get the neighbours up to the order on the grid of the given shape
        neighbours = fauvqe.utilities.generic.grid_neighbour_list(
            i,
            shape,
            neighbour_order,
            periodic=periodic,
            diagonal=diagonal,
            origin="topleft",
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
        neighbours = fauvqe.utilities.generic.grid_neighbour_list(
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
                    comb = list(fauvqe.utilities.generic.flatten(comb))
                    # not elegant but gotta avoid doubles
                    if (
                        i in comb
                        and comb not in all_combs
                        and list(reversed(comb)) not in all_combs
                    ):
                        term = fauvqe.utilities.fermion.even_excitation(
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
    shape = fauvqe.utilities.circuit.qubits_shape(model.flattened_qubits)
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


def filter_out_identity(psum):
    psum_out = []
    if isinstance(psum, cirq.PauliString):
        if fauvqe.utilities.circuit.pauli_str_is_identity(pstr=psum):
            raise ValueError(
                "Trying to filter out the remove the identity in a PauliString consisting only of the identity: {}".format(
                    psum
                )
            )
        else:
            return psum

    for pstr in psum:
        if not fauvqe.utilities.circuit.pauli_str_is_identity(pstr=pstr):
            psum_out.append(pstr)
    return cirq.PauliSum.from_pauli_strings(psum_out)


def exp_from_pauli_sum(pauli_sum: cirq.PauliSum, theta):
    psum_no_identity = filter_out_identity(pauli_sum)
    # PauliSumExponential takes cares of hermitian/anti-hermitian matters
    psum_exp = cirq.PauliSumExponential(pauli_sum_like=psum_no_identity, exponent=theta)
    return psum_exp
    # PauliSumExponential only accept A,B st exp(A)*exp(B) = exp(A+B) so might as well break them and "trotterize" them if they dont commute
    # return [cirq.PauliSumExponential(pauli_sum_like=pstr,exponent=theta) for pstr in pauli_sum if not pauli_str_is_identity(pstr)]


def preprocess_for_energy(ham: np.ndarray, op: np.ndarray):
    comm = np.matmul(ham, op)
    return comm


def preprocess_for_fidelity(op: np.ndarray, target_state: np.ndarray):
    # A * |gs>
    preprocessed_fid_state = op.dot(target_state)
    return preprocessed_fid_state


def compute_lowrank(
    ham: np.ndarray,
    op: np.ndarray,
    wf: np.ndarray,
):
    # this should be the lowrank method
    pass


def compute_preprocessed_fidelity_gradient(
    trial_state: np.ndarray, preprocessed_fid_state: np.ndarray
) -> float:
    # preprocessed_fid_op is e^(-theta A) A |gs>
    n_qubits = int(np.log2(trial_state.shape[0]))
    qid_shape = (2,) * n_qubits
    fidelity = cirq.fidelity(trial_state, preprocessed_fid_state, qid_shape=qid_shape)
    return fidelity


def compute_fidelity_gradient(
    op: np.ndarray,
    trial_state: np.ndarray,
    target_state: np.ndarray,
) -> float:
    # fidelity is |<psi|e^(-theta A)|gs>|
    # then df/dthetha = -<psi|Ae^(-theta A)|gs>
    #                 = -<psi|e^(-theta A)A|gs>
    # at theta=0, we have = -<psi|A|gs>
    ket = op.dot(target_state)
    n_qubits = int(np.log2(trial_state.shape[0]))
    qid_shape = (2,) * n_qubits

    fidelity = cirq.fidelity(-trial_state.conj(), ket, qid_shape=qid_shape)
    return fidelity


def compute_preprocessed_energy_gradient(
    trial_state: np.ndarray, preprocessed_energy_op: np.ndarray
) -> float:
    # dE/dt = 2Re<ps|HA|ps>
    ket = preprocessed_energy_op.dot(trial_state)
    bra = trial_state.conj()
    grad = np.abs(2 * np.real(np.dot(bra, ket)))
    return grad


def compute_energy_gradient(
    ham: np.ndarray,
    op: np.ndarray,
    trial_state: np.ndarray,
    full: bool = True,
    eps: float = 1e-5,
    sparse: bool = False,
) -> float:
    if sparse:
        if not full:
            # in the noiseless case dE/dt = 2Re<ps|exp(-theta A) HA exp(theta A)|ps> (eval at theta=0)
            spham = scipy.sparse.csc_matrix(ham)
            spop = scipy.sparse.csc_matrix(op)
            spwf = scipy.sparse.csc_matrix(trial_state)
            comm = spham @ spop
            ket = comm.dot(spwf.transpose())
            bra = spwf.conj()
            grad = float(np.abs(2 * np.real((bra @ ket).toarray())))
        else:
            # <p|exp(-eps*operator)[H,A(k)]exp(eps*operator)|p>
            # = <p|exp(-eps*operator)(HA(k) - A(k)H)exp(eps*operator)|p> = dE/dk
            # finite diff (f(theta + eps) - f(theta - eps))/ 2eps but theta = 0
            # if A is anti hermitian
            # (<p|exp(-eps*operator) H exp(eps*operator)|p> - <p|exp(eps*operator) H exp(-eps*operator)|p>)/2eps
            # if A is hermitian the theta needs to be multiplied by 1j
            wfexp = scipy.sparse.linalg.expm_multiply(
                A=op, B=trial_state, start=-eps, stop=eps, num=2, endpoint=True
            )
            spham = scipy.sparse.csc_matrix(ham)
            # exp(-theta A)|psi>
            spwf0 = scipy.sparse.csc_matrix(wfexp[0, :]).transpose()
            # # exp(theta A)|psi>
            spwf1 = scipy.sparse.csc_matrix(wfexp[1, :]).transpose()
            grad_minus = (spwf1.getH() @ spham @ spwf1).toarray()
            grad_plus = (spwf0.getH() @ spham @ spwf0).toarray()
            grad = float(np.abs((grad_minus - grad_plus) / (2 * eps)))
    # dense methods (get it?)
    else:
        if not full:
            # dE/dt = 2Re<ps|exp(-theta A) HA exp(theta A)|ps>
            comm = np.matmul(ham, op)
            grad = np.abs(2 * np.real(np.vdot(trial_state.T, comm.dot(trial_state))))
        else:
            # (<p|exp(-eps*operator) H exp(eps*operator)|p> - <p|exp(eps*operator) H exp(-eps*operator)|p>)/2eps
            wfexp = scipy.sparse.linalg.expm_multiply(
                A=op, B=trial_state, start=-eps, stop=eps, num=2, endpoint=True
            )
            wfexp_plus = wfexp[1, :]
            wfexp_minus = wfexp[0, :]
            grad_minus = np.vdot(wfexp_minus, ham @ wfexp_minus)
            grad_plus = np.vdot(wfexp_plus, ham @ wfexp_plus)
            grad = np.abs((grad_minus - grad_plus) / (2 * eps))

    return grad
