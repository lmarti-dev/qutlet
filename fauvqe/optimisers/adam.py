"""
This is the ADAM-submodule for Optimiser()

This file is not exectuded, rather called within Ising() class when:
-set_circuit('qaoa') is called 

or functions are handed over to classical optimiser

Implementation following arXiv 1412.6980

Algorithm 1: Adam, our proposed algorithm for stochastic optimization. See section 2 for details,
and for a slightly more efﬁcient (but less clear) order of computation. g 2
t indicates the elementwise square g_t  g_t . Good default settings for the tested machine learning problems are α = 0.001,
β_1 = 0.9, β_2 = 0.999 and eps = 10− 8 . All operations on vectors are element-wise. With β1 and β2 we denote β_1 and β_2 to the power t.

Require: α: Stepsize
Require: β1 , β2 ∈ [0, 1): Exponential decay rates for the moment estimates
Require: f(θ): Stochastic objective function with parameters θ
Require: θ 0 : Initial parameter vector
    m_0 ← 0 (Initialize 1 st moment vector)
    v_0 ← 0 (Initialize 2 nd moment vector)
    t ← 0   (Initialize timestep)
    while θ t not converged do
        t ← t + 1
        g_t ← ∇_θ f_t (θ_{t−1} )                    (Get gradients w.r.t. stochastic objective at timestep t)
        m_t ← β_1 · m_t + (1 − β_1 ) · g_t      (Update biased ﬁrst moment estimate)
        v_t ← β_2 · v_t + (1 − β_2 ) · g^2_t    (Update biased second raw moment estimate)
        \hat m_t ← m_t /(1 − β^t_1 )                (Compute bias-corrected ﬁrst moment estimate)
        \hat v_t ← v_t /(1 − β^t_2 )                (Compute bias-corrected second raw moment estimate)
        θ_t ← θ_{t−1} − α · \hat m_t /( (\hat v_t)^0.5 + eps) (Update parameters)
    end while
return θ t (Resulting parameters)

Potential MUCH BETTER alternative option: load from scipy??

"""
import numpy as np
import cirq
from numbers import Real, Integral
from typing import Union, Literal

from fauvqe.optimisers.optimiser import Optimiser
from fauvqe.optimisers.optimisation_step import OptimisationStep
from fauvqe.optimisers.optimisation_result import OptimisationResult
from fauvqe.objectives.objective import Objective


class ADAM(Optimiser):
    """
    Args from parrent class:
        obj_func()          :   objectiv function f(x)/energy
        qubits              :   qubit array/ordering for parametrised circuit
        simulator           :   Classical quantum simulator to simulate circuit
        circuit             :   parametrised circuit
        circuit_param       :   sympy.Symbols for parametrised circuit
        circuit_param_values:   current/initial values of circuit parameters
                                    ->To be updates

    Additional Args (give default values):
      eps         :eps for gradient
      eps_2       : epsilon for adam
      b_1, b_2    : beta_1, beta_2 for adam

      break_cond  :default 'iterations', but also e.g. change in obj_func etc.
      break_param : e.g amount of steps of iteration

    Try to include/implement MPI4py or multiprocessing here!
    """

    def __init__(
        self,
        eps: Real = 10 ** -3,
        eps_2: Real = 10 ** -8,
        a: Real = 10 ** -2,
        b_1: Real = 0.9,
        b_2: Real = 0.999,
        break_cond: Literal["iterations"] = "iterations",
        break_param: Integral = 100,
    ):
        super().__init__()
        assert all(
            isinstance(n, Real) and n > 0.0 for n in [eps, eps_2, a, b_1, b_2]
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
        self._eps: Real = eps
        self._eps_2: Real = eps_2
        self._a: Real = a
        self._b_1: Real = b_1
        self._b_2: Real = b_2
        self._break_cond: Literal["iterations"] = break_cond
        self._break_param: Integral = break_param

        # The following attributes change for each objective
        self._objective: Union[Objective, None] = None
        self._v_t: np.ndarray = np.array([])
        self._m_t: np.ndarray = np.array([])
        self._circuit_param: np.ndarray = np.array([])
        self._n_param: Integral = 0

    def optimise(self, objective: Objective) -> OptimisationResult:
        """Run optimiser until break condition is fulfilled

        1. Make copies of param_values (to not accidentally overwrite)
        2. Do steps until break condition. Tricky need to generalise _get_param_resolver method which also affects _get_gradients
        3. Update self.circuit_param_values = temp_cpv

        Parameters
        ----------
        objective: Objective
            The objective to optimise

        Returns
        -------
        OptimisationResult

        Raises
        -------
        AssertionError
        NotImplementedError
        """
        assert isinstance(
            objective, Objective
        ), "objective is not an instance of a subclass of Objective, given type '{}'".format(
            type(objective).__name__
        )
        self._objective = objective
        res = OptimisationResult(objective)

        self._circuit_param = objective.initialiser.circuit_param

        # 1.make copies of param_values (to not accidentally overwrite)
        temp_cpv = objective.initialiser.circuit_param_values.view()
        self._n_param = np.size(temp_cpv)
        self._v_t = np.zeros(np.shape(temp_cpv))
        self._m_t = np.zeros(np.shape(temp_cpv))

        # Do step until break condition
        if self._break_cond == "iterations":
            for i in range(self._break_param):
                # Make ADAM step
                temp_cpv = self._ADAM_step(temp_cpv, step=i + 1)
                res.add_step(temp_cpv.copy())

        return res

    def _ADAM_step(self, temp_cpv, step: int):
        """
        t ← t + 1
        g_t ← ∇_θ f_t (θ_{t−1} )                    (Get gradients w.r.t. stochastic objective at timestep t)
        m_t ← β_1 · m_t + (1 − β_1 ) · g_t      (Update biased ﬁrst moment estimate)
        v_t ← β_2 · v_t + (1 − β_2 ) · g^2_t    (Update biased second raw moment estimate)
        \hat m_t ← m_t /(1 − β^t_1 )                (Compute bias-corrected ﬁrst moment estimate)
        \hat v_t ← v_t /(1 − β^t_2 )                (Compute bias-corrected second raw moment estimate)
        θ_t ← θ_{t−1} − α · \hat m_t /( (\hat v_t)^0.5 + eps) (Update parameters)

        Alternative for last 3 lines:
        α_t = α · (1 − β^t_2)^0.5/(1 − β^t_1)
        θ_t ← θ_{t−1} − α_t · m_t  /( (v_t)^0.5 + eps)

        """
        gradient_values = self._get_gradients(temp_cpv)
        self._m_t = self._b_1 * self._m_t + (1 - self._b_1) * gradient_values
        self._v_t = self._b_2 * self._v_t + (1 - self._b_2) * gradient_values ** 2
        temp_cpv -= (
            self._a
            * (1 - self._b_2 ** step) ** 0.5
            / (1 - self._b_1 ** step)
            * self._m_t
            / (self._v_t ** 0.5 + self._eps_2)
        )
        return temp_cpv

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
