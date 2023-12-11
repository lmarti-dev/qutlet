from qutlet.utilities import default_value_handler

from scipy.optimize import OptimizeResult, minimize
from typing import Dict, Iterable, Union
from qutlet.circuits.ansatz import Ansatz

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
        minimize_options={"method": "L-BFGS-B"},
        initial_state=None,
        method_options: dict = {},
        save_sim_data: bool = True,
    ):
        self._minimize_options = {}
        self._minimize_options.update(minimize_options)
        self._method_options = {}
        self._method_options.update(method_options)
        self.initial_state = initial_state
        self._function_calls_count = 0
        self.save_sim_data = save_sim_data
        if save_sim_data:
            self.sim_data = []

            def add_step(**kwargs):
                self.sim_data.append(kwargs)

            self.add_step: callable = add_step

            def fun(self, x):
                sim_result = self.ansatz.simulate(x, initial_state=self.initial_state)
                objective_value = self._objective(sim_result)
                self.add_step(sim_result=sim_result, objective_value=objective_value)
                self._function_calls_count += 1
                return objective_value

        else:

            def fun(self, x):
                self._function_calls_count += 1
                sim_result = self.ansatz.simulate(x, initial_state=self.initial_state)
                objective_value = self._objective(sim_result)

                return objective_value

        self.fun = fun

    def optimise(
        self,
        ansatz: Ansatz,
        objective: callable,
        initial_params: Union[str, float, Iterable],
    ) -> OptimizeResult:
        self._objective = objective
        self.ansatz = ansatz
        x0 = default_value_handler(
            shape=(ansatz.n_symbols,),
            value=initial_params,
        )
        result: OptimizeResult = minimize(
            self.fun, x0, **self._minimize_options, options=self._method_options
        )
        print("function calls: ", self._function_calls_count)
        if self.save_sim_data:
            return result, self.sim_data
        else:
            return result

    def __to_json__(self) -> Dict:
        return {
            "constructor_params": {
                "save_each_function_call": self.save_each_function_call,
                "minimize_options": self._minimize_options,
                "method_options": self._method_options,
                "initial_state": self.initial_state,
            },
        }

    @classmethod
    def from_dict(cls, dct) -> Dict:
        return cls(**dct["constructor_params"])
