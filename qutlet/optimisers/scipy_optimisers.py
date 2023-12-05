from qutlet.utilities.generic import default_value_handler
from qutlet.utilities.circuit import get_param_resolver
from qiskit.algorithms.optimizers.nft import nakanishi_fujii_todo


from scipy.optimize import OptimizeResult, minimize
import numpy as np
from typing import Dict, Iterable, Union
import cirq
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
    ):
        self._minimize_options = {}
        self._minimize_options.update(minimize_options)
        self._method_options = {}
        self._method_options.update(method_options)
        self.initial_state = initial_state
        self._function_calls_count = 0
        super().__init__()

    def fun(self, x):
        self._function_calls_count += 1
        state = self.ansatz.simulate(x, initial_state=self.initial_state)
        objective_value = self._objective(state=state)
        return objective_value

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
        scipy_res = minimize(
            self.fun, x0, **self._minimize_options, options=self._method_options
        )
        print("function calls: ", self._function_calls_count)
        return scipy_res

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "save_each_function_call": self.save_each_function_call,
                "minimize_options": self._minimize_options,
                "method_options": self._method_options,
                "initial_state": self.initial_state,
            },
        }

    @classmethod
    def from_json_dict(cls, dct) -> Dict:
        return cls(**dct["constructor_params"])
