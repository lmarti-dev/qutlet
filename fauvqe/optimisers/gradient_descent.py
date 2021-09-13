"""
This is a submodule of GradientOptimiser()

This file is not exectuded, rather called within Ising() class when:
-set_circuit('qaoa') is called

or functions are handed over to classical optimiser
"""
from numbers import Real, Integral
from typing import Literal, Union, Dict, Optional, List

import cirq
import numpy as np

from fauvqe.objectives.objective import Objective
from fauvqe.optimisers.optimisation_result import OptimisationResult
from fauvqe.optimisers.gradientoptimiser import GradientOptimiser


class GradientDescent(GradientOptimiser):
    """GradientDescent implementation as a GradientOptimiser.

    Arguments
    -----------
    optimiser_options: dict
            Dictionary containing additional options to individualise the optimisation routine. Contains:
                eps : Real default 0.001
                    Discretisation finesse for numerical gradient
                
                eta : Real default 0.01
                    Step size for parameter update rule
                
                break_cond : {"iterations"} default "iterations"
                    Break condition for optimisation
                
                break_param : int default 100
                    Amount of steps of iteration
                
                batch_size: Integral
                    number of batch wavefunctions, une None as initial_state if batch_size = 0 
            
                symmetric_gradient: bool
                    Specifies whether to use symmetric numerical gradient or asymmetric gradient (faster by ~ factor 2)
                
                plot_run: bool
                    Plot cost development in optimisation run and save to fauvqe/plots
                
                use_progress_bar: bool
                    Determines whether to use tqdm's progress bar when running the optimisation
    """

    def __init__(
        self,
        options: dict = {}
    ):
        super().__init__(options)
        if(self.options['batch_size'] > 0):
            self._optimise_step = self._optimise_step_indices
    
    def _optimise_step(self, temp_cpv: np.ndarray, _n_jobs: Integral, step: Integral):
        """
        Run optimiser until break condition is fullfilled

        1. make copies of param_values (to not accidentially overwrite)
        2. Do steps until break condition.
        3. Update self.circuit_param_values = temp_cpv
        """
        gradient_values, cost = self._get_gradients(temp_cpv, _n_jobs)
        temp_cpv -= self.options['eta'] * gradient_values
        return temp_cpv, cost
    
    def _optimise_step_indices(self, temp_cpv: np.ndarray, _n_jobs: Integral, step: Integral, indices: Optional[List[int]] = None):
        gradient_values, cost = self._get_gradients(temp_cpv, _n_jobs, indices)
        temp_cpv -= self.options['eta'] * gradient_values
        return temp_cpv, cost