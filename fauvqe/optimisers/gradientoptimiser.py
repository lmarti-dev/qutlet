"""
This abstract class resembles functions and attributes which Gradient Optimiser have in common. It hence inherits from Optimiser().

"""
import multiprocessing
from numbers import Real, Integral
from typing import Literal, Optional, Dict

import cirq
import joblib
import numpy as np
import abc

from fauvqe.objectives.objective import Objective
from fauvqe.optimisers.optimisation_result import OptimisationResult
from fauvqe.optimisers.optimiser import Optimiser


class GradientOptimiser(Optimiser):
    """Abstract GradientOptimiser class"""

    def __init__(
        self,
        eps: Real = 1e-3,
        eta: Real = 1e-2,
        break_cond: Literal["iterations", "accuracy"] = "iterations",
        break_param: Integral = 100,
        break_tol: Real = 1e-12
    ):
        """GradientOptimiser
        
        Parameters
        ----------
        eps: Real
            Discretisation finesse for numerical gradient
        eta: Real
            Step size for parameter update rule
        break_cond: {"iterations", "accuracy"} default "iterations"
            Break condition for the optimisation
        break_param: Integral
            "iterations" break parameter for the optimisation
        break_tol: Real
            "accuracy" break parameter for the optimisation
        """

        super().__init__()
        assert all(
            isinstance(n, Real) and n > 0.0 for n in [eps, eta, break_tol]
        ), "Parameters must be positive, real numbers"
        assert (
            isinstance(break_param, Integral) and break_param > 0
        ), "Break parameter must be a positive integer"
        valid_break_cond = ["iterations", "accuracy"]
        assert (
            break_cond in valid_break_cond
        ), "Invalid break condition, received: '{}', allowed are {}".format(
            break_cond, valid_break_cond
        )

        # The following attributes remain constant for the lifetime of this optimiser
        self._eps: Real = eps
        self._eta: Real = eta
        self._break_cond: Literal["iterations"] = break_cond
        self._break_param: Integral = break_param
        self._break_tol: Real = break_tol

        # The following attributes change for each objective
        self._objective: Optional[Objective] = None
        self._circuit_param: np.ndarray = np.array([])
        self._n_param: Integral = 0

    def optimise(
        self,
        objective: Objective,
        continue_at: Optional[OptimisationResult] = None,
        n_jobs: Integral = -1,
        initial_state: Optional[np.ndarray] = None,
    ) -> OptimisationResult:
        """Run optimiser until break condition is fulfilled

        1. Make copies of param_values (to not accidentally overwrite)
        2. Do steps until break condition. Tricky need to generalise get_param_resolver method
        3. Update self.circuit_param_values = temp_cpv

        Parameters
        ----------
        n_jobs: Integral, default -1
            The number ob simultaneous jobs (-1 for auto detection)
        objective: Objective
            The objective to optimise
        continue_at: OptimisationResult, optional
            Continue a optimisation
        initial_state: np.ndarray, optional
            Specify an initial state for objective.simulate

        Returns
        -------
        OptimisationResult
        """
        assert isinstance(
            objective, Objective
        ), "objective is not an instance of a subclass of Objective, given type '{}'".format(
            type(objective).__name__
        )
        self._objective = objective
        self._circuit_param = objective.model.circuit_param
        if continue_at is not None:
            assert isinstance(
                continue_at, OptimisationResult
            ), "continue_at must be a OptimisationResult"
            res = continue_at
            params = continue_at.get_latest_step().params
            temp_cpv = params.view()
            skip_steps = len(continue_at.get_steps())
        else:
            res = OptimisationResult(objective)
            temp_cpv = objective.model.circuit_param_values.view()
            skip_steps = 0

        self._n_param = np.size(temp_cpv)
        
        # Handle n_jobs parameter
        assert isinstance(
            n_jobs, Integral
        ), "The number of jobs must be a positive integer or -1 (default)'. Given: {}".format(
            n_jobs
        )

        # Determine maximal number of threads and reset qsim 't' flag for n_job = -1 (default)
        if n_jobs < 1:
            # max(n_jobs) = 2*n_params, as otherwise overhead of not used jobs
            n_jobs = int(
                min(
                    max(multiprocessing.cpu_count() / 2, 1),
                    2 * np.size(self._circuit_param),
                )
            )
            assert n_jobs != 0, "{} {}".format(
                multiprocessing.cpu_count(), np.size(self._circuit_param)
            )
            # Try to reset qsim threads, which overwrites the simulator if it was not qsim
            try:
                if str(self._objective.model.simulator.__class__).find('qsim') > 0:
                    sim_name = "qsim"
                t_new = int(max(np.divmod(multiprocessing.cpu_count() / 2, n_jobs)[0], 1))
                self._objective.model.set_simulator(simulator_name=sim_name,simulator_options={"t": t_new})
            except:
                pass

        # Do step until break condition
        if self._break_cond == "iterations":
            for i in range(self._break_param - skip_steps):
                temp_cpv = self._cpv_update(temp_cpv, n_jobs, step=i + 1, initial_state = initial_state)
                res.add_step(temp_cpv.copy())
        elif(self._break_cond == "accuracy"):
            raise NotImplementedError()
        return res

    @abc.abstractmethod
    def _cpv_update(self, temp_cpv: np.ndarray, _n_jobs: Integral, step: Integral, initial_state: Optional[np.ndarray] = None):
        """
        Perform Optimiser specific update rule and return new parameters
        
        Parameters
        ----------
        temp_cpv: np.ndarray        current parameters
        _n_jobs: Integral        Number of jobs to rum parallel
        step: Integral        number of current step
        """
        raise NotImplementedError()
    
    def _get_gradients(self, temp_cpv, _n_jobs, initial_state: Optional[np.ndarray] = None):
        joined_dict = {**{str(self._circuit_param[i]): temp_cpv[i] for i in range(self._n_param)}}
        # backend options: -'loky'               seems to be always the slowest
        #                   'multiprocessing'   crashes where both other options do not
        #                   'threading'         supposedly best option, still so far slower than
        #                                       seqquential _get_gradients()
        # 1. Get single energies via joblib.Parallel
        # Format E_1 +eps, E_1 - eps
        # Potential issue: Need 2*n_param copies of joined_dict
        _energies = joblib.Parallel(n_jobs=_n_jobs, backend="loky")(
            joblib.delayed(self._get_single_cost)(joined_dict.copy(), j, initial_state)
            for j in range(2 * self._n_param)
        )
        # 2. Return gradiens
        # Make np array?
        _energies = np.array(_energies).reshape((self._n_param, 2))
        gradients_values = np.matmul(_energies, np.array((1, -1))) / (2 * self._eps)
        return gradients_values

    def _get_single_cost(self, joined_dict, j, initial_state: Optional[np.ndarray] = None):
        # Simulate wavefunction at p_j +/- eps
        # j even: +  j odd: -

        # Alternative: _str_cp = str(self.circuit_param[np.divmod(j,2)[0]])
        joined_dict[str(self._circuit_param[np.divmod(j, 2)[0]])] = (
            joined_dict[str(self._circuit_param[np.divmod(j, 2)[0]])]
            + 2 * (0.5 - np.mod(j, 2)) * self._eps
        )
        wf = self._objective.simulate(param_resolver=cirq.ParamResolver(joined_dict), initial_state=initial_state)

        return self._objective.evaluate(wf)

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "eps": self._eps,
                "eta": self._eta,
                "break_cond": self._break_cond,
                "break_param": self._break_param,
                "break_tol": self._break_tol,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])
