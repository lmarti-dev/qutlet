"""
This is a submodule for Optimiser()

This file is not exectuded, rather called within Ising() class when:
-set_circuit('qaoa') is called

or functions are handed over to classical optimiser
"""
from numbers import Real, Integral
from typing import Literal, Union

import numpy as np
import cirq

from fauvqe.optimisers.optimiser import Optimiser
from fauvqe.optimisers.optimisation_result import OptimisationResult
from fauvqe.objectives.objective import Objective


class GradientDescent(Optimiser):
    """GradientDescent implementation as an Optimiser.

    Arguments
    -----------
    eps : Real default 0.001
      Epsilon for gradient

    eta : Real default 0.01
      Velocity of gradient method

    break_cond : {"iterations"} default "iterations"
      Break condition for optimisation

    break_param : int default 100
      Amount of steps of iteration

    n_print : int default -1
      debug print out after n steps, disable with -1
    """

    def __init__(
        self,
        eps: Real = 10 ** -3,
        eta: Real = 10 ** -2,
        break_cond: Literal["iterations"] = "iterations",
        break_param: Integral = 100,
    ):
        super().__init__()
        assert all(
            isinstance(n, Real) and n > 0.0 for n in [eps, eta]
        ), "Parameters must be positive, real numbers"
        assert (
            isinstance(break_param, Integral) and break_param > 0
        ), "Break parameter must be a positive integer"
        valid_break_cond = ["iterations"]
        assert (
            break_cond in valid_break_cond
        ), "Invalid break condition, received: '{}', allowed are {}".format(
            break_cond, valid_break_cond
        )

        # The following attributes remain constant for the lifetime of this optimiser
        self._eps = eps
        self._eta = eta
        self._break_cond = break_cond
        self._break_param = break_param

        # The following attributes change for each objective
        self._objective: Union[Objective, None] = None
        self._circuit_param: np.ndarray = np.array([])
        self._n_param: Integral = 0

    def optimise(self, objective: Objective) -> OptimisationResult:
        """
        Run optimiser until break condition is fullfilled

        1. make copies of param_values (to not accidentially overwrite)
        2. Do steps until break condition.
        3. Update self.circuit_param_values = temp_cpv
        """
        assert isinstance(
            objective, Objective
        ), "objective is not an instance of a subclass of Objective, given type '{}'".format(
            type(objective).__name__
        )
        self._objective = objective

        res = OptimisationResult(objective)

        self._circuit_param = objective.model.circuit_param

        # 1.make copies of param_values (to not accidentally overwrite)
        temp_cpv = objective.model.circuit_param_values.view()
        self._n_param = np.size(temp_cpv)

        # Do step until break condition
        if self._break_cond == "iterations":
            for _ in range(self._break_param):
                gradient_values = self._get_gradients(temp_cpv)

                # Make gradient step
                # Potentially improve this:
                for j in range(np.size(temp_cpv)):
                    temp_cpv[j] -= self._eta * gradient_values[j]

                res.add_step(temp_cpv.copy())

        return res

    def _get_gradients(self, temp_cpv):
        gradient_values = np.zeros(self._n_param)

        # Create joined dictionary from
        # self.circuit_param <- need their name list
        #     create name array?
        # Use temp_cpv not self.circuit_param_values here
        joined_dict = {**{str(self._circuit_param[i]): temp_cpv[i] for i in range(self._n_param)}}

        # Calculate the gradients
        # Use MPi4Py/Multiprocessing HERE!!
        for j in range(self._n_param):
            # Simulate wavefunction at p_j + eps
            joined_dict[str(self._circuit_param[j])] = (
                joined_dict[str(self._circuit_param[j])] + self._eps
            )
            wf1 = self._objective.simulate(param_resolver=cirq.ParamResolver(joined_dict))

            # Simulate wavefunction at p_j - eps
            joined_dict[str(self._circuit_param[j])] = (
                joined_dict[str(self._circuit_param[j])] - 2 * self._eps
            )
            wf2 = self._objective.simulate(param_resolver=cirq.ParamResolver(joined_dict))

            # Calculate gradient
            gradient_values[j] = (self._objective.evaluate(wf1) - self._objective.evaluate(wf2)) / (
                2 * self._eps
            )

            # Reset dictionary
            joined_dict[str(self._circuit_param[j])] = (
                joined_dict[str(self._circuit_param[j])] + self._eps
            )

        return gradient_values
