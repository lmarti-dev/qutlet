import scipy
from fauvqe.optimisers.optimiser import Optimiser
from fauvqe.objectives.objective import Objective
from fauvqe.optimisers.optimisation_result import OptimisationResult
from fauvqe.utilities.generic import default_value_handler
from fauvqe.utilities.circuit import get_param_resolver

from scipy.optimize import minimize, OptimizeResult
import numpy as np
from typing import Dict, Iterable, Union
import cirq


# available optimizers:
# ‘Nelder-Mead’
# ‘Powell’
# ‘CG’
# ‘BFGS’
# ‘Newton-CG’
# ‘L-BFGS-B’
# ‘TNC’
# ‘COBYLA’
# ‘SLSQP’
# ‘trust-constr’
# ‘dogleg’
# ‘trust-ncg’
# ‘trust-exact’
# ‘trust-krylov’


class ScipyOptimisers(Optimiser):
    def __init__(
        self,
        minimize_options={"method": "L-BFGS-B"},
        initial_state=None,
        save_each_function_call: bool = False,
        method_options: dict = {},
    ):
        self.save_each_function_call = save_each_function_call
        self._minimize_options = {}
        self._minimize_options.update(minimize_options)
        self._method_options = {}
        self._method_options.update(method_options)
        self.initial_state = initial_state
        self._function_calls_count = 0
        super().__init__()

    def simulate(self, x):
        wf = self._objective.simulate(
            param_resolver=get_param_resolver(
                model=self._objective.model, param_values=x
            ),
            initial_state=self.initial_state,
        )
        return wf

    def fun(self, x):
        self._function_calls_count += 1
        wf = self.simulate(x)
        objective_value = self._objective.evaluate(wavefunction=wf)
        if self.save_each_function_call:
            self._fauvqe_res.add_step(
                params=x, wavefunction=wf, objective=objective_value
            )
        return objective_value

    def optimise(
        self, objective: Objective, initial_params: Union[str, float, Iterable]
    ):
        self._objective = objective
        self._fauvqe_res = OptimisationResult(self._objective)
        x0 = default_value_handler(
            shape=np.shape(self._objective.model.circuit_param_values.view()),
            value=initial_params,
        )

        def process_step(xk):
            wf = self.simulate(xk)
            objective_value = self._objective.evaluate(wavefunction=wf)
            self._fauvqe_res.add_step(
                params=xk, wavefunction=wf, objective=objective_value
            )

        # add the initial step
        process_step(x0)
        if not self.save_each_function_call:
            callback = process_step
        else:
            callback = None
        scipy_res = minimize(
            self.fun,
            x0,
            **self._minimize_options,
            callback=callback,
            options=self._method_options
        )
        # add the last step when the simulation is done
        x_final = scipy_res.x
        process_step(x_final)
        print("function calls: ", self._function_calls_count)
        return self._fauvqe_res

    def from_json_dict(self) -> Dict:
        pass

    def to_json_dict(self) -> Dict:
        pass
