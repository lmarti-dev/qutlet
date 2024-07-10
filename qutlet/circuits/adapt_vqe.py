import multiprocessing as mp
from functools import partial
from typing import Union, Iterable
import cirq
import numpy as np
import scipy
import sympy

from qutlet.circuits.adapt_gate_pools import (
    GatePool,
    ExponentiableGatePool,
    SubCircuitSet,
    PauliStringSet,
    HamiltonianPauliSumSet,
    FermionicPauliSumSet,
    PauliSumListSet,
    PauliSumSet,
    exp_from_pauli_sum,
)
from qutlet.models import QubitModel

# necessary for the full loop
from qutlet.optimisers.scipy_optimisers import ScipyOptimisers
from qutlet.utilities import (
    depth,
    get_param_resolver,
    match_param_values_to_symbols,
    normalize_vec,
    populate_empty_qubits,
    identity_on_qubits,
)


class ADAPT:
    def __init__(
        self,
        model: QubitModel,
        gate_pool_options: Union[dict, list],
        measure: dict,
        preprocess: bool,
        verbosity: int = 0,
        n_jobs: int = 1,
    ):
        self.model = model
        self.measure = {"name": None, "target_state": None}
        self.measure.update(measure)
        # if verbosity is true or false cast to int
        self.verbosity = int(verbosity)
        self.preprocess = preprocess
        self.n_jobs = n_jobs
        self.indices_to_ignore = []
        if isinstance(gate_pool_options, list):
            self.pool_name = "custom"
            self.gate_pool = gate_pool_options
        elif isinstance(gate_pool_options, dict):
            self.pool_name = gate_pool_options.pop("pool_name", None)
            self.gate_pool_class: GatePool = self.pick_gate_pool(
                pool_name=self.pool_name, gate_pool_options=gate_pool_options
            )
            self.gate_pool = self.gate_pool_class.operator_pool
        self.preprocessed_ops = [None] * len(self.gate_pool)
        if self.preprocess:
            self.preprocessed_ops = self.preprocess_gate_gradients()

    def gate_pool_op_ind_to_dispatch(self, ind: int):
        if isinstance(self.gate_pool_class, ExponentiableGatePool):
            return self.gate_pool_class.matrix(ind=ind, qubits=self.model.qubits)
        elif isinstance(self.gate_pool_class, SubCircuitSet):
            return self.gate_pool_class.operator_pool[ind]
        else:
            raise ValueError("how did you get that far without an error?")

    def measure_gradient(self, ind: int, trial_state: np.ndarray):
        self.verbose_print("computing {g} gradient".format(g=self.gate_pool[ind]))
        ham = self.model.hamiltonian.matrix(qubits=self.model.qubits)
        grad = measure_gradient_dispatcher(
            measure_name=self.measure["name"],
            preprocess=self.preprocess,
            trial_state=trial_state,
            preprocessed_op=self.preprocessed_ops[ind],
            ham=ham,
            operator=self.gate_pool_op_ind_to_dispatch(ind),
            target_state=self.measure["target_state"],
            qubits=self.model.qubits,
        )
        self.verbose_print("gradient: {:.10f}".format(grad))
        return grad

    def verbose_print(self, s: str, message_level: int = 1):
        if int(self.verbosity) >= message_level:
            print(s)

    def pick_gate_pool(self, pool_name, gate_pool_options):
        set_options = {}
        if pool_name == "pauli_string":
            set_options.update({"qubits": self.model.qubits})
            set_options.update(gate_pool_options)
            return PauliStringSet(**set_options)
        elif pool_name == "hamiltonian":
            set_options.update({"threshold": None})
            set_options.update(gate_pool_options)
            return HamiltonianPauliSumSet(model=self.model, **set_options)
        elif pool_name == "fermionic":
            return FermionicPauliSumSet(model=self.model, set_options=gate_pool_options)
        elif pool_name == "pauli_sums":
            set_options.update({"qubits": self.model.qubits})
            set_options.update(gate_pool_options)
            return PauliSumSet(**set_options)
        elif pool_name == "pauli_lists":
            set_options.update({"qubits": self.model.qubits})
            set_options.update(gate_pool_options)
            return PauliSumListSet(**set_options)
        elif pool_name == "sub_circuit":
            set_options.update({"qubits": self.model.qubits})
            set_options.update(gate_pool_options)
            return SubCircuitSet(**set_options)
        else:
            raise ValueError("expected a valid pool name, got {}".format(pool_name))

    def preprocess_gate_gradients(self):
        self.verbose_print(
            "Preprocessing {n} gates of {p} pool for {m}, preprocessing: {b}".format(
                n=len(self.gate_pool),
                p=self.pool_name,
                m=self.measure["name"],
                b=self.preprocess,
            )
        )

        preprocess_ops = []

        if self.measure["name"] == "energy":
            preprocess_options = {
                "ham": self.model.hamiltonian.matrix(qubits=self.model.qubits)
            }
            preprocess_fn = preprocess_for_energy
        elif self.measure["name"] == "fidelity":
            preprocess_options = {
                "target_state": self.measure["target_state"],
            }
            preprocess_fn = preprocess_for_fidelity
        if self.n_jobs > 1:
            self.verbose_print(
                "Starting {n} jobs for parallel preprocessing".format(n=self.n_jobs)
            )
            process_pool = mp.Pool(self.n_jobs)
            results = process_pool.map(
                partial(preprocess_fn, **preprocess_options),
                [op.matrix(qubits=self.model.qubits) for op in self.gate_pool],
            )
            # should the process pool be in init and kept throughout the instance life?
            process_pool.close()
            process_pool.join()
            for result in results:
                # pool.map result are ordered the same way as the input
                preprocess_ops.append(result)

        else:
            for op in self.gate_pool:
                op_mat = op.matrix(qubits=self.model.qubits)
                preprocess_ops.append(preprocess_fn(op=op_mat, **preprocess_options))
        self.verbose_print(
            "Preprocessing finished, {} preprocessed ops".format(len(preprocess_ops))
        )
        return preprocess_ops

    def get_max_gradient(
        self,
        tol: float,
        trial_state: np.ndarray = None,
        initial_state: np.ndarray = None,
    ):
        # we can set the trial wf (before computing the gradient of the energy)
        # if we want some specific psi to apply the gate to.
        # not very useful for now, might come in handy later.
        # initial_state is the state to input in the circuit
        if trial_state is None:
            circ = populate_empty_qubits(model=self.model)
            trial_state = self.model.simulator.simulate(
                circ,
                param_resolver=get_param_resolver(
                    model=self.model, param_values=self.model.circuit_param_values
                ),
                initial_state=initial_state,
            ).final_state_vector

        grad_values = []
        self.verbose_print(
            "Gate pool size: {}".format(
                len(self.gate_pool) - len(self.indices_to_ignore)
            )
        )
        self.verbose_print(
            "measure chosen: {}, preprocessing: {}".format(
                self.measure["name"], self.preprocess
            )
        )
        if self.n_jobs > 1:
            ham = self.model.hamiltonian.matrix(qubits=self.model.qubits)
            process_pool = mp.Pool(self.n_jobs, maxtasksperchild=1000)
            self.verbose_print(
                "Pooling {} jobs to compute gradient".format(self.n_jobs)
            )

            if self.preprocess:
                results = process_pool.starmap(
                    measure_gradient_dispatcher,
                    [
                        (
                            self.gate_pool_op_ind_to_dispatch(ind),
                            self.measure["name"],
                            self.preprocess,
                            trial_state,
                            self.preprocessed_ops[ind],
                            ham,
                            self.measure["target_state"],
                            self.model.qubits,
                        )
                        for ind in range(len(self.gate_pool))
                    ],
                    chunksize=32,
                )
            else:
                partial_dispatcher = partial(
                    measure_gradient_dispatcher,
                    measure_name=self.measure["name"],
                    preprocess=False,
                    trial_state=trial_state,
                    preprocessed_op=None,
                    ham=ham,
                    target_state=self.measure["target_state"],
                    qubits=self.model.qubits,
                )
                results = process_pool.imap(
                    partial_dispatcher,
                    (
                        self.gate_pool_op_ind_to_dispatch(ind)
                        for ind in range(len(self.gate_pool))
                    ),
                    chunksize=32,
                )
            # should the process pool be in init and kept?
            process_pool.close()
            process_pool.join()
            grad_values = [grad for grad in results]

        else:
            for ind in range(len(self.gate_pool)):
                grad = self.measure_gradient(ind=ind, trial_state=trial_state)
                grad_values.append(grad)

        if not self.indices_to_ignore:
            max_index = np.argmax(np.abs(grad_values))
        else:
            # in case we want to ignore some operators, we take the largest indice that we don't ignore
            sorted_indices = (-np.array(np.abs(grad_values))).argsort()
            max_index = [
                ind for ind in sorted_indices if ind not in self.indices_to_ignore
            ][0]
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
            return max_index

    def gate_and_param_from_pool(self, ind):
        param_name = param_name_from_circuit(circ=self.model.circuit)
        gate, theta = self.gate_pool_class.gate_from_op(ind=ind, param_name=param_name)
        self.verbose_print("Using gate: {gate}".format(gate=gate))
        return gate, theta

    def append_best_gate_to_circuit(
        self,
        tol: float = 1e-15,
        trial_state: np.ndarray = None,
        initial_state: np.ndarray = None,
    ) -> int:
        max_index = self.get_max_gradient(
            tol=tol,
            trial_state=trial_state,
            initial_state=initial_state,
        )
        if max_index is None:
            self.verbose_print("No best gate found or threshold reached, exiting")
            return None
        else:
            gates, theta = self.gate_and_param_from_pool(ind=max_index)
            for gate in gates:
                self.model.circuit += gate
            if isinstance(theta, Iterable):
                self.model.circuit_param.extend(theta)
            else:
                self.model.circuit_param.append(theta)
            match_param_values_to_symbols(
                model=self.model,
                symbols=self.model.circuit_param,
                default_value="zeros",
            )
            self.verbose_print(
                "best gate found and added: {best_gate}".format(best_gate=gates)
            )
            return max_index

    # pick random gate for benchmarking
    def append_random_gate_to_circuit(
        self,
        default_value: str = "random",
    ):
        ind = np.random.choice(len(self.gate_pool))
        param_name = param_name_from_circuit(circ=self.model.circuit)
        theta = sympy.Symbol("theta_{param_name}".format(param_name=param_name))
        exp_gates = exp_from_pauli_sum(pauli_sum=self.gate_pool[ind], theta=theta)
        for gate in exp_gates:
            self.model.circuit += gate
        self.model.circuit_param.append(theta)
        match_param_values_to_symbols(
            model=self.model,
            symbols=self.model.circuit_param,
            default_value=default_value,
        )
        self.verbose_print("adding random gate: {rgate}".format(rgate=exp_gates))
        return False

    def loop(
        self,
        n_steps: int,
        optimiser,
        objective,
        tetris: bool = False,
        discard_previous_best: Union[bool, int] = False,
        trial_state: np.ndarray = None,
        initial_state: np.ndarray = None,
        random: bool = False,
        callback: callable = None,
    ):
        result = None
        if isinstance(discard_previous_best, int):
            # how many steps to wait until we can reuse a gate
            reuse_countdown = discard_previous_best
        else:
            reuse_countdown = -1
        if tetris and discard_previous_best:
            # this is the tetris-adapt-vqe thing where you put gates on unused qubits
            # can't have both
            raise ValueError("can't both discard gates and use tetris procedure")

        for step in range(n_steps):
            if random:
                # in case you want to benchmark adapt by comparing it with a random circuit
                self.append_random_gate_to_circuit("zeros")
                max_index = -1
            else:
                # this returns the best index to be added to the blacklist
                max_index = self.append_best_gate_to_circuit(
                    trial_state=trial_state, initial_state=initial_state
                )

            # if the gradient is above the threshold, go on
            if max_index is not None:
                self.verbose_print("optimizing {}-th step".format(step + 1))
                # optimize to get a result everytime
                self.model.circuit = populate_empty_qubits(model=self.model)

                if isinstance(optimiser, ScipyOptimisers):
                    # this will only work with the scipy optimiser, as setting the initial guess for the parameters
                    result = optimiser.optimise(
                        objective=objective,
                        initial_params="random",
                    )
                else:
                    result = optimiser.optimise(objective=objective)
                self.model.circuit_param_values = result.get_latest_step().params
                print(
                    "circuit depth: {d}, number of params: {p}".format(
                        d=depth(self.model.circuit),
                        p=len(self.model.circuit_param),
                    )
                )
                # callback every step so that we can process each optimisation round
                if callback is not None:
                    callback(result, step)
                # if we want to forbid adapt from adding the same gate every time (this happens sometimes)
                if tetris:
                    circuit_last_moment = self.model.circuit[-1]
                    qubits = circuit_last_moment.qubits
                    overlapping_indices = [
                        ind
                        for ind in range(len(self.gate_pool))
                        if set(qubits) & set((q for q in self.gate_pool[ind]))
                    ]
                    print(
                        f"{len(overlapping_indices)}/{len(self.gate_pool)} qubit-overlapping gates"
                    )
                    if len(overlapping_indices) < len(self.gate_pool):
                        self.indices_to_ignore = overlapping_indices
                    else:
                        self.indices_to_ignore = []
                if discard_previous_best:
                    # if we reached the number of steps we've forbidden a gate to be used for, reset the blacklist
                    if reuse_countdown == 0:
                        self.indices_to_ignore = []
                        reuse_countdown = discard_previous_best
                        self.verbose_print("resetting blacklist")
                    self.indices_to_ignore.append(max_index)
                    reuse_countdown -= 1
            else:
                if tetris and len(self.indices_to_ignore) > 0:
                    # try again perhaps
                    print(
                        "tetris game over: {}/{} gates are being ignored".format(
                            len(self.indices_to_ignore), len(self.gate_pool)
                        )
                    )
                    self.indices_to_ignore = []
                else:
                    print("treshold reached, exiting")
                    break

        return result


# ugly method, but necessary for multiprocessing
def measure_gradient_dispatcher(
    operator: Union[np.ndarray, cirq.Circuit],
    measure_name: str,
    preprocess: bool,
    trial_state: np.ndarray,
    preprocessed_op: np.ndarray,
    ham: np.ndarray,
    target_state: np.ndarray,
    qubits: list[cirq.Qid],
):
    eps = 1e-5
    # for subcirc
    if isinstance(operator, cirq.Circuit):
        eps = 1
        if measure_name == "energy":
            return circuit_energy_gradient(
                ham=ham,
                circuit=operator,
                qubits=qubits,
                trial_state=trial_state,
                eps=eps,
            )
        elif measure_name == "fidelity":
            return circuit_fidelity_gradient(
                trial_state=trial_state,
                qubits=qubits,
                circuit=operator,
                target_state=target_state,
                eps=eps,
            )
    # for the exponentiable ops
    else:
        if measure_name == "energy":
            if preprocess:
                grad = compute_preprocessed_energy_gradient(
                    trial_state=trial_state,
                    preprocessed_energy_op=preprocessed_op,
                )
            else:
                grad = compute_energy_gradient(
                    ham=ham,
                    op=operator,
                    trial_state=trial_state,
                    full=True,
                    eps=eps,
                    sparse=False,
                )
        elif measure_name == "fidelity":
            if preprocess:
                grad = compute_preprocessed_fidelity_gradient(
                    trial_state=trial_state,
                    preprocessed_fid_state=preprocessed_op,
                )
            else:
                grad = compute_fidelity_gradient(
                    op=operator,
                    trial_state=trial_state,
                    target_state=target_state,
                )
        else:
            raise ValueError("measure_name: {} unknown".format(measure_name))
    return grad


def preprocess_for_energy(op: np.ndarray, ham: np.ndarray):
    comm = np.matmul(ham, op)
    return comm


def preprocess_for_fidelity(op: np.ndarray, target_state: np.ndarray):
    # A * |gs>
    preprocessed_fid_state = op.dot(target_state)
    return preprocessed_fid_state


def param_name_from_circuit(circ: cirq.Circuit, proptype="ops") -> str:
    if proptype == "ops":
        num = sum(1 for _ in circ.all_operations())
    elif proptype == "depth":
        num = depth(circuit=circ)
    return "p_" + str(num)


def compute_low_rank_energy_gradient(
    ham: np.ndarray,
    op: np.ndarray,
    wf: np.ndarray,
):
    # TODO: this should be the lowrank method
    pass


def compute_preprocessed_fidelity_gradient(
    trial_state: np.ndarray, preprocessed_fid_state: np.ndarray
) -> float:
    # preprocessed_fid_op is e^(-theta A) A |gs>
    n_qubits = int(np.log2(trial_state.shape[0]))
    qid_shape = (2,) * n_qubits
    fidelity = cirq.fidelity(trial_state, preprocessed_fid_state, qid_shape=qid_shape)
    return fidelity**2


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
    if np.linalg.norm(ket) == 0:
        return 0
    elif np.linalg.norm(ket) != 1:
        ket = normalize_vec(ket)
    n_qubits = int(np.log2(trial_state.shape[0]))
    qid_shape = (2,) * n_qubits

    fidelity = cirq.fidelity(-trial_state.conj(), ket, qid_shape=qid_shape)
    return fidelity**2


def compute_preprocessed_energy_gradient(
    trial_state: np.ndarray, preprocessed_energy_op: np.ndarray
) -> float:
    # dE/dt = 2Re<ps|HA|ps>
    ket = preprocessed_energy_op.dot(trial_state)
    bra = trial_state.conj()
    grad = np.abs(2 * np.real(np.dot(bra, ket)))
    return grad


def evolve_trial_state(
    circuit: cirq.Circuit, qubits: list[cirq.Qid], trial_state: np.ndarray, eps: float
):
    param_res_pos = cirq.ParamResolver(
        {k: eps for k in cirq.parameter_symbols(circuit)}
    )
    populated_circuit = identity_on_qubits(circuit=circuit, qubits=qubits)
    final_state = populated_circuit.final_state_vector(
        initial_state=trial_state,
        param_resolver=param_res_pos,
    )
    return final_state


def circuit_energy_gradient(
    ham: np.ndarray,
    circuit: cirq.Circuit,
    qubits: list[cirq.Qid],
    trial_state: np.ndarray,
    eps: float,
):
    final_state_pos = evolve_trial_state(
        circuit=circuit, qubits=qubits, trial_state=trial_state, eps=eps
    )
    final_state_neg = evolve_trial_state(
        circuit=circuit, qubits=qubits, trial_state=trial_state, eps=-eps
    )

    exp_pos = np.real(np.vdot(final_state_pos, ham @ final_state_pos))
    exp_neg = np.real(np.vdot(final_state_neg, ham @ final_state_neg))
    return np.abs((exp_pos - exp_neg) / (2 * eps))


def circuit_fidelity_gradient(
    trial_state: np.ndarray,
    circuit: cirq.Circuit,
    qubits: list[cirq.Qid],
    target_state: np.ndarray,
    eps: float,
):
    qid_shape = (2,) * int(np.log2(len(trial_state)))

    final_state_pos = evolve_trial_state(
        circuit=circuit, qubits=qubits, trial_state=trial_state, eps=eps
    )
    final_state_neg = evolve_trial_state(
        circuit=circuit, qubits=qubits, trial_state=trial_state, eps=-eps
    )
    fid_pos = cirq.fidelity(final_state_pos, target_state, qid_shape=qid_shape)
    fid_neg = cirq.fidelity(final_state_neg, target_state, qid_shape=qid_shape)
    return np.abs((fid_pos - fid_neg) / (2 * eps))


def compute_energy_gradient(
    ham: np.ndarray,
    op: np.ndarray,
    trial_state: np.ndarray,
    full: bool = True,
    eps: float = 1e-5,
    sparse: bool = False,
) -> float:
    # fastest is full non sparse, so those are the default
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
