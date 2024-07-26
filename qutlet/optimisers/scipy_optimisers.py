from qutlet.utilities import default_value_handler

from scipy.optimize import OptimizeResult, minimize
from typing import Dict, Iterable, Union
from qutlet.circuits.ansatz import Ansatz
import numpy as np

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


class ScipyOptimisers:
    def __init__(
        self,
        ansatz: Ansatz,
        objective: callable,
        minimize_options={"method": "L-BFGS-B"},
        method_options: dict = {},
        save_sim_data: bool = True,
        callback: callable = None,
    ):
        self._objective = objective
        self.ansatz = ansatz
        self._minimize_options = {}
        self._minimize_options.update(minimize_options)
        self._method_options = {}
        self._method_options.update(method_options)
        self._function_calls_count = 0
        self.save_sim_data = save_sim_data
        # callback(xk) for COBYLA
        # callback(intermediate_result: OptimizeResult) for others
        self.callback = callback
        if save_sim_data:
            self.sim_data = {}

            def add_step(**kwargs):
                for k in kwargs.keys():
                    if k not in self.sim_data:
                        self.sim_data[k] = []
                    else:
                        self.sim_data[k].append(kwargs[k])

        else:

            def add_step(**kwargs):
                return None

        self.add_step: callable = add_step

        def fun(x: np.ndarray, initial_state: np.ndarray):
            sim_result = self.ansatz.simulate(opt_params=x, initial_state=initial_state)
            objective_value = self._objective(sim_result)
            self.add_step(sim_result=sim_result, objective_value=objective_value)
            self._function_calls_count += 1
            return objective_value

        self.fun = fun

    def optimise(
        self,
        initial_params: Union[str, float, Iterable],
        initial_state: np.ndarray = None,
    ) -> OptimizeResult:
        x0 = default_value_handler(
            shape=(self.ansatz.n_symbols,),
            value=initial_params,
        )
        result: OptimizeResult = minimize(
            self.fun,
            x0,
            initial_state,
            **self._minimize_options,
            options=self._method_options,
            callback=self.callback,
        )
        print("function calls: ", self._function_calls_count)
        if self.save_sim_data:
            return result, self.sim_data
        else:
            return result, None

    def __to_json__(self) -> Dict:
        return {
            "constructor_params": {
                "ansatz": self.ansatz,
                "objective": self._objective,
                "minimize_options": self._minimize_options,
                "method_options": self._method_options,
            },
            "sim_data": self.sim_data,
        }

    @classmethod
    def from_dict(cls, dct) -> Dict:
        return cls(**dct["constructor_params"])
