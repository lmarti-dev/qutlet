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
from numbers import Real, Integral
from typing import Union, Literal
import multiprocessing

import numpy as np
import cirq
import joblib

from fauvqe.optimisers.optimiser import Optimiser
from fauvqe.optimisers.optimisation_result import OptimisationResult
from fauvqe.objectives.objective import Objective


class ADAM(Optimiser):
    """ADAM class docstring"""

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
        """

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

    def optimise( self, objective: Objective, n_jobs: Union[Integral] = -1) -> OptimisationResult:
        """Run optimiser until break condition is fulfilled. Use n_jobs = 1 to essentially previous non-parallel version of optimise()
        Watch out: Due to initialisation cost of n_jobs = 2 might be more expensive than n_jobs = 1


        1. Make copies of param_values (to not accidentally overwrite)
        2. Do steps until break condition. Tricky need to generalise get_param_resolver method
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
        NotImpl
        """
        self._objective = objective
        self._circuit_param = objective.model.circuit_param

        # Determine maximal number of threads and reset qsim 't' flag for n_job = -1 (default)
        if n_jobs < 1:
            # max(n_jobs) = 2*n_params, as otherwise overhead of not used jobs
            n_jobs = int(min(max(multiprocessing.cpu_count() / 2,1), 2*np.size(self._circuit_param)))
            # Try to reset qsim threads, which overrights the simulator if it was not qsim
            try: 
                t_new = int(max(np.divmod(multiprocessing.cpu_count() / 2, n_jobs)[0],1))
                self._objective.model.set_simulator(simulator_options={'t' : t_new})
            except:
                pass

        assert isinstance( n_jobs, Integral), \
        "The number of jobs must be an integer or 'default'. Given: {}".format(n_jobs)

        print("n_jobs: \t {}".format(n_jobs))

        assert isinstance(objective, Objective), \
        "objective is not an instance of a subclass of Objective, given type '{}'".format(type(objective).__name__)

        # 1.make copies of param_values (to not accidentally overwrite)
        temp_cpv = objective.model.circuit_param_values.view()
        self._n_param = np.size(temp_cpv)
        self._v_t = np.zeros(np.shape(temp_cpv))
        self._m_t = np.zeros(np.shape(temp_cpv))

        res = OptimisationResult(objective)
        # Do step until break condition
        if self._break_cond == "iterations":
            for step in range(self._break_param):
                # Make ADAM step
                temp_cpv = self._ADAM_step(temp_cpv, n_jobs, step=step + 1)
                res.add_step(temp_cpv.copy())

        return res

    def _ADAM_step(self, temp_cpv: np.ndarray, _n_jobs: Integral, step: Integral):
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
        gradient_values = np.array(self._get_gradients(temp_cpv, _n_jobs))
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

    def _get_gradients(self, temp_cpv, _n_jobs):
        joined_dict = {**{str(self._circuit_param[i]): temp_cpv[i] for i in range(self._n_param)}}
        # backend options: -'loky'               seems to be always the slowest
        #                   'multiprocessing'   crashes where both other options do not
        #                   'threading'         supposedly best option, still so far slower than
        #                                       seqquential _get_gradients()
        # 1. Get single energies via joblib.Parallel
        # Format E_1 +eps, E_1 - eps
        # Potential issue: Need 2*n_param copies of joined_dict
        _energies = joblib.Parallel(n_jobs=_n_jobs, backend="loky")(
            joblib.delayed(self._get_single_energy)(joined_dict.copy(), j)
            for j in range(2 * self._n_param)
        )
        # 2. Return gradiens
        # Make np array?
        _energies = np.array(_energies).reshape((self._n_param, 2))
        gradients_values = np.matmul(_energies, np.array((1, -1))) / (2 * self._eps)
        return gradients_values

    def _get_single_energy(self, joined_dict, j):
        # Simulate wavefunction at p_j +/- eps
        # j even: +  j odd: -

        # Alternative: _str_cp = str(self.circuit_param[np.divmod(j,2)[0]])
        joined_dict[str(self._circuit_param[np.divmod(j, 2)[0]])] = (
            joined_dict[str(self._circuit_param[np.divmod(j, 2)[0]])]
            + 2 * (0.5 - np.mod(j, 2)) * self._eps
        )
        wf = self._objective.simulate(param_resolver=cirq.ParamResolver(joined_dict))

        return self._objective.evaluate(wf)
