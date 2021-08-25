"""
This is the ADAM-submodule for Optimiser()

This file is not exectuded, rather called within Ising() class when:
-set_circuit('qaoa') is called

or functions are handed over to classical optimiser

Implementation following arXiv 1412.6980

Algorithm 1: Adam, our proposed algorithm for stochastic optimization. See section 2 for details,
and for a slightly more efficient (but less clear) order of computation. g 2
t indicates the element wise square g_t  g_t .
Good default settings for the tested machine learning problems are α = 0.001,
β_1 = 0.9, β_2 = 0.999 and eps = 10− 8 . All operations on vectors are element-wise.
With β1 and β2 we denote β_1 and β_2 to the power t.

Require: α: Step size
Require: β1 , β2 ∈ [0, 1): Exponential decay rates for the moment estimates
Require: f(θ): Stochastic objective function with parameters θ
Require: θ 0 : Initial parameter vector
    m_0 ← 0 (Initialize 1 st moment vector)
    v_0 ← 0 (Initialize 2 nd moment vector)
    t ← 0   (Initialize time step)
    while θ t not converged do
        t ← t + 1
        g_t ← ∇_θ f_t (θ_{t−1} )  (Get gradients w.r.t. stochastic objective at time step t)
        m_t ← β_1 · m_t + (1 − β_1 ) · g_t      (Update biased first moment estimate)
        v_t ← β_2 · v_t + (1 − β_2 ) · g^2_t    (Update biased second raw moment estimate)
        \\hat m_t ← m_t /(1 − β^t_1 ) (Compute bias-corrected first moment estimate)
        \\hat v_t ← v_t /(1 − β^t_2 ) (Compute bias-corrected second raw moment estimate)
        θ_t ← θ_{t−1} − α · \\hat m_t /( (\\hat v_t)^0.5 + eps) (Update parameters)
    end while
return θ t (Resulting parameters)

Potential MUCH BETTER alternative option: load from scipy??

"""
import multiprocessing
from numbers import Real, Integral
from typing import Literal, Optional, Dict, List

import cirq
import joblib
import timeit
import numpy as np

from fauvqe.objectives.objective import Objective
from fauvqe.optimisers.optimisation_result import OptimisationResult
from fauvqe.optimisers.gradientoptimiser import GradientOptimiser


class ADAM(GradientOptimiser):
    """ADAM class docstring"""

    def __init__(
        self,
        eps: Real = 1e-3,
        eps_2: Real = 1e-8,
        eta: Real = 1e-2,
        b_1: Real = 0.9,
        b_2: Real = 0.999,
        break_cond: Literal["iterations"] = "iterations",
        break_param: Integral = 100,
        break_tol: Real = 1e-12,
        batch_size: Integral = 0,
        symmetric_gradient: bool = True,
    ):
        """ADAM optimiser

        Parameters
        ----------
        eps: Real
            :math:`\\epsilon` for gradient
        eps_2: Real
            :math:`\\epsilon` for adam
        a: Real
            Step size :math:`\\alpha` for adam
        b_1: Real
            :math:`\\beta_1` for adam
        b_2: Real
            :math:`\\beta_2` for adam
        break_cond: {"iterations"} default "iterations"
            Break condition for the optimisation
        break_param: Integral
            Break parameter for the optimisation
        symmetric_gradient: bool
            Specifies whether to use symmetric numerical gradient or asymmetric gradient (faster by ~ factor 2)
        """

        super().__init__(eps, eta, break_cond, break_param, break_tol, batch_size, symmetric_gradient)
        assert all(
            isinstance(n, Real) and n > 0.0 for n in [eps_2, b_1, b_2]
        ), "Parameters must be positive, real numbers"
        
        # The following attributes remain constant for the lifetime of this optimiser
        self._eps_2: Real = eps_2
        self._b_1: Real = b_1
        self._b_2: Real = b_2
        
        # The following attributes change for each objective
        self._objective: Optional[Objective] = None
        self._v_t: np.ndarray = np.array([])
        self._m_t: np.ndarray = np.array([])
        self._circuit_param: np.ndarray = np.array([])
        self._n_param: Integral = 0

    def optimise(
        self,
        objective: Objective,
        continue_at: Optional[OptimisationResult] = None,
        n_jobs: Integral = -1,
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

        Returns
        -------
        OptimisationResult
        """
        #Do we even need these two lines?
        self._v_t = np.zeros(np.shape(objective.model.circuit_param_values.view()))
        self._m_t = np.zeros(np.shape(objective.model.circuit_param_values.view()))
        return super().optimise(objective, continue_at, n_jobs)
        
    def _cpv_update(self, temp_cpv: np.ndarray, _n_jobs: Integral, step: Integral, indices: Optional[List[int]] = None):
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
        gradient_values = self._get_gradients(temp_cpv, _n_jobs, indices)
        self._m_t = self._b_1 * self._m_t + (1 - self._b_1) * gradient_values
        self._v_t = self._b_2 * self._v_t + (1 - self._b_2) * gradient_values ** 2
        temp_cpv -= (
            self._eta
            * (1 - self._b_2 ** step) ** 0.5
            / (1 - self._b_1 ** step)
            * self._m_t
            / (self._v_t ** 0.5 + self._eps_2)
        )
        
        return temp_cpv
    
    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "eps": self._eps,
                "eps_2": self.eps_2,
                "eta": self._eta,
                "b_1": self._b_1,
                "b_2": self._b_2,
                "break_cond": self._break_cond,
                "break_param": self._break_param,
                "batch_size": self._batch_size
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])