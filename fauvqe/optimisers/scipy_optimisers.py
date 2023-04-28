import scipy
from fauvqe.optimisers.optimiser import Optimiser
from fauvqe.objectives.objective import Objective
from fauvqe.optimisers.optimisation_result import OptimisationResult
import fauvqe.utils as utils
import fauvqe.utils_cirq as cqutils

from scipy.optimize import minimize, OptimizeResult
import numpy as np
from typing import Dict
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
        initial_params="random",
        save_each_function_call: bool = False,
        method_options: dict = {},
    ):
        self.save_each_function_call = save_each_function_call
        self.minimize_options = minimize_options
        self.method_options = {}
        self.method_options.update(method_options)
        self.initial_state = initial_state
        self.initial_params = initial_params
        self.function_calls_count = 0
        super().__init__()

    def simulate(self, x):
        wf = self._objective.simulate(
            param_resolver=cqutils.get_param_resolver(model=self._objective.model, param_values=x),
            initial_state=self.initial_state,
        )
        return wf

    def fun(self, x):
        self.function_calls_count += 1
        wf = self.simulate(x)
        objective_value = self._objective.evaluate(wavefunction=wf)
        if self.save_each_function_call:
            self._fauvqe_res.add_step(params=x, wavefunction=wf, objective=objective_value)
        return objective_value

    def optimise(self, objective: Objective):
        self._objective = objective
        self._fauvqe_res = OptimisationResult(self._objective)
        x0 = utils.default_value_handler(
            shape=np.shape(self._objective.model.circuit_param_values.view()),
            value=self.initial_params,
        )

        def process_step(xk):
            wf = self.simulate(xk)
            objective_value = self._objective.evaluate(wavefunction=wf)
            self._fauvqe_res.add_step(params=xk, wavefunction=wf, objective=objective_value)

        # add the initial step
        process_step(x0)
        if not self.save_each_function_call:
            callback = process_step
        else:
            callback = None
        scipy_res = minimize(
            self.fun, x0, **self.minimize_options, callback=callback, options=self.method_options
        )
        # add the last step when the simulation is done
        x_final = scipy_res.x
        process_step(x_final)
        print("function calls: ", self.function_calls_count)
        return self._fauvqe_res

    def from_json_dict(self) -> Dict:
        pass

    def to_json_dict(self) -> Dict:
        pass
