"""
This is a submodule for GradientOptimiser()

This file is not exectuded, rather called within Ising() class when:
-set_circuit('qaoa') is called

or functions are handed over to classical optimiser
"""
from numbers import Real, Integral
from typing import Literal, Union, Dict

import cirq
import numpy as np

from fauvqe.objectives.objective import Objective
from fauvqe.optimisers.optimisation_result import OptimisationResult
from fauvqe.optimisers.gradientoptimiser import GradientOptimiser


class GradientDescent(GradientOptimiser):
    """GradientDescent implementation as a GradientOptimiser.

    Arguments
    -----------
    eps : Real default 0.001
      Discretisation finesse for numerical gradient

    eta : Real default 0.01
      Step size for parameter update rule

    break_cond : {"iterations", "accuracy"} default "iterations"
      Break condition for optimisation

    break_param : int default 100
      Amount of steps of iteration
    
    break_tol: Real default 1e-12
      "accuracy" break parameter for the optimisation
    """

    def __init__(
        self,
        eps: Real = 1e-3,
        eta: Real = 1e-2,
        break_cond: Literal["iterations", "accuracy"] = "iterations",
        break_param: Integral = 100,
        break_tol: Real = 1e-12
    ):
        super().__init__(eps, eta, break_cond, break_param, break_tol)

    def _cpv_update(self, temp_cpv: np.ndarray, _n_jobs: Integral, step: Integral):
        """
        Run optimiser until break condition is fullfilled

        1. make copies of param_values (to not accidentially overwrite)
        2. Do steps until break condition.
        3. Update self.circuit_param_values = temp_cpv
        """
        return temp_cpv - self._eta * gradient_values

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "eps": self._eps,
                "eta": self._eta,
                "break_cond": self._break_cond,
                "break_param": self._break_param,
                "break_tol": self._break_tol
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])
