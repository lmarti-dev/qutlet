import multiprocess as mp
from functools import partial
from typing import Union, Iterable
import cirq
import numpy as np
import sympy

from qutlet.circuits.adapt_gate_pools import (
    GatePool,
    exp_from_pauli_sum,
)
from qutlet.circuits.ansatz import Ansatz, param_name_from_circuit
from qutlet.models import QubitModel

# necessary for the full loop
from qutlet.optimisers.scipy_optimisers import ScipyOptimisers
from qutlet.utilities import (
    depth,
    match_param_values_to_symbols,
    populate_empty_qubits,
    identity_on_qubits,
)

# dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
# mp.reduction.ForkingPickler = dill.Pickler
# mp.reduction.dump = dill.dump


class ADAPT(Ansatz):
    def __init__(
        self,
        model: QubitModel,
        gate_pool: GatePool,
        objective: callable,
        verbosity: int = 0,
        n_jobs: int = 1,
    ):
        super().__init__()
        self.model = model
        self.objective = objective
        # if verbosity is true or false cast to int
        self.verbosity = int(verbosity)
        self.n_jobs = n_jobs
        self.indices_to_ignore = []

        self.gate_pool: GatePool = gate_pool
        self.operator_pool = self.gate_pool.operator_pool

        self.process_pool = None
        if self.n_jobs > 1:

            self.process_pool = mp.Pool(self.n_jobs)

    def verbose_print(self, s: str, message_level: int = 1, end="\n"):
        if int(self.verbosity) >= message_level:
            print(s, end=end)

    def get_max_gradient(
        self,
        tol: float,
        initial_state: np.ndarray = None,
    ):

        trial_state = self.simulate(
            initial_state=initial_state, state_qubits=self.model.qubits
        )

        grad_values = []
        self.verbose_print(
            "Gate pool size: {}".format(
                len(self.operator_pool) - len(self.indices_to_ignore)
            )
        )

        indices = [
            x for x in range(len(self.operator_pool)) if x not in self.indices_to_ignore
        ]

        if self.n_jobs > 1:
            self.verbose_print(
                "Pooling {} jobs to compute gradient".format(self.n_jobs)
            )

            partial_dispatcher = partial(
                measure_gradient_dispatcher,
                trial_state=trial_state,
                qubits=self.model.qubits,
                objective=self.objective,
            )
            gates = (
                cirq.Circuit(self.gate_pool.gate_from_op(ind, "dummy")[0])
                for ind in indices
            )

            results = self.process_pool.map(
                partial_dispatcher,
                gates,
                chunksize=32,
            )

            grad_values = [grad for grad in results]

        else:

            for ind in indices:
                grad = measure_gradient_dispatcher(
                    trial_state=trial_state,
                    gate=cirq.Circuit(self.gate_pool.gate_from_op(ind, "dummy")[0]),
                    objective=self.objective,
                    qubits=self.model.qubits,
                )
                self.verbose_print(
                    f"{ind}/{len(self.operator_pool)} gradient: {grad:.5f}", end="\r"
                )
                grad_values.append(grad)

        # in case we want to ignore some operators, we take the largest indice that we don't ignore
        max_index = indices[np.argmax(np.abs(grad_values))]
        max_grad = np.max(np.abs(grad_values))
        if tol is not None and max_grad < tol:
            self.verbose_print(
                "gradient ({grad:.2f}) is smaller than tol ({tol}), exiting".format(
                    grad=max_grad, tol=tol
                )
            )
            # iteration process is done, gradient < tol, or if stopping is set with max depth, continue
            return None
        else:
            self.verbose_print(
                "gradient ({grad}) is larger than tolerance: ({tol}), continuing".format(
                    grad=max_grad, tol=tol
                ),
            )
            return max_index

    def gate_and_param_from_pool(self, ind):
        param_name = param_name_from_circuit(circ=self.circuit)
        gate, theta = self.gate_pool.gate_from_op(ind=ind, param_name=param_name)
        return gate, theta

    def append_best_gate_to_circuit(
        self,
        tol: float = 1e-15,
        initial_state: np.ndarray = None,
    ) -> int:
        max_index = self.get_max_gradient(
            tol=tol,
            initial_state=initial_state,
        )
        if max_index is None:
            self.verbose_print("No best gate found or threshold reached, exiting")
            return None
        else:
            gates, theta = self.gate_and_param_from_pool(ind=max_index)
            for gate in gates:
                self.circuit += gate
            if isinstance(theta, Iterable):
                self.symbols.extend(theta)
            else:
                self.symbols.append(theta)
            self.params, self.symbols = match_param_values_to_symbols(
                params=self.params,
                symbols=self.symbols,
                default_value="random",
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
        ind = np.random.choice(len(self.operator_pool))
        param_name = param_name_from_circuit(circ=self.circuit)
        theta = sympy.Symbol("theta_{param_name}".format(param_name=param_name))
        exp_gates = exp_from_pauli_sum(pauli_sum=self.operator_pool[ind], theta=theta)
        for gate in exp_gates:
            self.circuit += gate
        self.symbols.append(theta)
        self.params, self.symbols = match_param_values_to_symbols(
            params=self.params,
            symbols=self.symbols,
            default_value=default_value,
        )
        self.verbose_print("adding random gate: {rgate}".format(rgate=exp_gates))
        return False

    def loop(
        self,
        n_steps: int,
        optimiser: ScipyOptimisers,
        tetris: bool = False,
        discard_previous_best: Union[bool, int] = False,
        initial_state: np.ndarray = None,
        random: bool = False,
        callback: callable = None,
        threshold: float = 1e-10,
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
                self.append_random_gate_to_circuit("random")
                max_index = -1
            else:
                # this returns the best index to be added to the blacklist
                max_index = self.append_best_gate_to_circuit(
                    initial_state=initial_state
                )

            # if the gradient is above the threshold, go on
            if max_index is not None:
                self.verbose_print("optimizing {}-th step".format(step + 1))
                # optimize to get a result everytime
                self.circuit = populate_empty_qubits(
                    circuit=self.circuit, qubits=self.model.qubits
                )

                if isinstance(optimiser, ScipyOptimisers):
                    # this will only work with the scipy optimiser, as setting the initial guess for the parameters
                    if optimiser.save_sim_data is False:
                        raise ValueError("Must enable save_sim_data")

                    result, sim_data = optimiser.optimise(
                        initial_params=self.params, initial_state=initial_state
                    )
                else:
                    raise ValueError("Only works with ScipyOptimisers")
                self.params = result["x"]
                self.verbose_print(
                    "circuit depth: {d}, number of params: {p}".format(
                        d=depth(self.circuit),
                        p=len(self.params),
                    )
                )
                if sim_data["objective_value"][-1] <= threshold:
                    break
                # callback every step so that we can process each optimisation round
                if callback is not None:
                    callback(result, sim_data, step)
                # if we want to forbid adapt from adding the same gate every time (this happens sometimes)
                if tetris:
                    circuit_last_moment = self.circuit[-1]
                    qubits = circuit_last_moment.qubits
                    overlapping_indices = [
                        ind
                        for ind in range(len(self.operator_pool))
                        if set(qubits)
                        & set((q for q in self.operator_pool[ind].qubits))
                    ]
                    self.verbose_print(
                        f"{len(overlapping_indices)}/{len(self.operator_pool)} qubit-overlapping gates"
                    )
                    if len(overlapping_indices) < len(self.operator_pool):
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
                    self.verbose_print(
                        "tetris game over: {}/{} gates are being ignored".format(
                            len(self.indices_to_ignore), len(self.operator_pool)
                        )
                    )
                    self.indices_to_ignore = []
                else:
                    self.verbose_print("treshold reached, exiting")
                    break

        if self.process_pool is not None:
            # should the process pool be in init and kept?
            self.process_pool.close()
            self.process_pool.terminate()
            self.process_pool.join()

        return result


# ugly method, but necessary for multiprocessing
def measure_gradient_dispatcher(
    gate: cirq.Circuit,
    trial_state: np.ndarray,
    objective: callable,
    qubits: list[cirq.Qid],
):
    # parameter shift rule
    eps = 1e-5
    # for subcirx
    # gate is circuit because mp
    return circuit_gradient(
        circuit=gate,
        qubits=qubits,
        trial_state=trial_state,
        eps=eps,
        objective=objective,
    )


def evolve_trial_state(
    circuit: cirq.Circuit, qubits: list[cirq.Qid], trial_state: np.ndarray, eps: float
):
    param_res = cirq.ParamResolver({k: eps for k in cirq.parameter_symbols(circuit)})
    populated_circuit = identity_on_qubits(circuit=circuit, qubits=qubits)
    final_state = populated_circuit.final_state_vector(
        initial_state=trial_state,
        param_resolver=param_res,
    )
    return final_state


def circuit_gradient(
    circuit: cirq.Circuit,
    qubits: list[cirq.Qid],
    trial_state: np.ndarray,
    eps: float,
    objective: callable,
):
    final_state_pos = evolve_trial_state(
        circuit=circuit, qubits=qubits, trial_state=trial_state, eps=eps
    )
    final_state_neg = evolve_trial_state(
        circuit=circuit, qubits=qubits, trial_state=trial_state, eps=-eps
    )

    val_pos = objective(final_state_pos)
    val_neg = objective(final_state_neg)

    return np.abs((val_pos - val_neg) / (2 * eps))
