"""
This abstract class resembles functions and attributes which Gradient Optimiser have in common. It hence inherits from Optimiser().

"""
from __future__ import annotations
import abc
import cirq

from fauvqe.objectives.objective import Objective
from fauvqe.optimisers.optimisation_result import OptimisationResult
from fauvqe.optimisers.optimiser import Optimiser

import importlib

import joblib

import math
import matplotlib.pyplot as plt
import multiprocessing
from numbers import Real, Integral
import numpy as np
import os
dir_path = os.path.abspath(os.path.dirname(__file__))
from sys import stdout
from typing import Literal, Optional, Dict, List


class GradientOptimiser(Optimiser):
    """Abstract GradientOptimiser class"""

    def __init__(
        self,
        options: dict = {}
    ):
        """GradientOptimiser
        
        Parameters
        ----------
        optimiser_options: dict
            Dictionary containing additional options to individualise the optimisation routine. Contains:
                eps: Real
                    Discretisation finesse for numerical gradient
                
                eta: Real
                    Step size for parameter update rule
                
                break_cond: {"iterations"} default "iterations"
                    Break condition for the optimisation
                
                break_param: Integral
                    "iterations" break parameter for the optimisation
                
                batch_size: Integral
                    number of batch wavefunctions, une None as initial_state if batch_size = 0 
                
                symmetric_gradient: bool
                    Specifies whether to use symmetric numerical gradient or asymmetric gradient (faster by ~ factor 2)
                
                plot_run: bool
                    Plot cost development in optimisation run and save to fauvqe/plots
                
                use_progress_bar: bool
                    Determines whether to use tqdm's progress bar when running the optimisation
        """

        super().__init__()
        
        # Update Optimiser Options
        self.options = {
            'eps': 1e-3, 
            'eta': 1e-2, 
            'break_cond': "iterations", 
            'break_param': 100, 
            'batch_size': 0, 
            'symmetric_gradient': True, 
            'plot_run': False, 
            'use_progress_bar': False
        }
        self.options.update(options)
        if(self.options['symmetric_gradient'] and self.options['batch_size'] == 0):
            self._get_gradients = self._get_gradients_sym
            self._get_single_cost = self._get_single_cost_sym
        elif((not self.options['symmetric_gradient']) and self.options['batch_size'] == 0):
            self._get_gradients = self._get_gradients_asym
            self._get_single_cost = self._get_single_cost_asym
        elif(self.options['symmetric_gradient'] and self.options['batch_size'] > 0):
            self._get_gradients = self._get_gradients_sym_indices
            self._get_single_cost = self._get_single_cost_sym_indices
        else:
            self._get_gradients = self._get_gradients_asym_indices
            self._get_single_cost = self._get_single_cost_asym_indices
        
        assert all(
            isinstance(n, Real) and n > 0.0 for n in [self.options['eps'], self.options['eta']]
        ), "Parameters must be positive, real numbers"
        assert (
            isinstance(n, Integral) and n > 0 for n in [self.options['break_param'], self.options['batch_size']]
        ), "Parameters must be positive integers"
        valid_break_cond = ["iterations"]
        assert (
            self.options['break_cond'] in valid_break_cond
        ), "Invalid break condition, received: '{}', allowed are {}".format(
            self.options['break_cond'], valid_break_cond
        )
        
        if(self.options['use_progress_bar']):
            self._tqdm = importlib.import_module("tqdm").tqdm
        
        # The following attributes change for each objective
        self._objective: Optional[Objective] = None
        self._circuit_param: np.ndarray = np.array([])
        self._n_param: Integral = 0

    def _set_default_n_jobs(self):
        # max(n_jobs) = 2*n_params, as otherwise overhead of not used jobs
        if(self.options['symmetric_gradient']):
            max_thread = 2 * np.size(self._circuit_param)
        else:
            max_thread = np.size(self._circuit_param) + 1
        n_jobs = int(
            min(
                max(multiprocessing.cpu_count() / 2, 1),
                max_thread,
            )
        )
        assert n_jobs != 0, "{} {}".format(
            multiprocessing.cpu_count(), np.size(self._circuit_param)
        )
        # Try to reset qsim threads, which overwrites the simulator if it was not qsim
        try:
            if str(self._objective.model.simulator.__class__).find('qsim') > 0:
                sim_name = "qsim"
            t_new = int(max(np.divmod(multiprocessing.cpu_count() / 2, n_jobs)[0], 1))
            self._objective.model.set_simulator(simulator_name=sim_name,simulator_options={"t": t_new})
        except:
            pass
        return n_jobs
    
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
        assert isinstance(
            objective, Objective
        ), "objective is not an instance of a subclass of Objective, given type '{}'".format(
            type(objective).__name__
        )
        self._objective = objective
        self._circuit_param = objective.model.circuit_param
        if continue_at is not None:
            assert isinstance(
                continue_at, OptimisationResult
            ), "continue_at must be a OptimisationResult"
            res = continue_at
            params = continue_at.get_latest_step().params
            temp_cpv = params.view()
            skip_steps = len(continue_at.get_steps())
        else:
            res = OptimisationResult(objective)
            temp_cpv = objective.model.circuit_param_values.view()
            skip_steps = 0

        self._n_param = np.size(temp_cpv)
        
        # Handle n_jobs parameter
        assert isinstance(
            n_jobs, Integral
        ), "The number of jobs must be a positive integer or -1 (default)'. Given: {}".format(
            n_jobs
        )

        # Determine maximal number of threads and reset qsim 't' flag for n_job = -1 (default)
        if n_jobs < 1:
            n_jobs = self._set_default_n_jobs()
        
        # Do step until break condition
        if self.options['break_cond'] == "iterations":
            #Set progress bar, if wanted
            if(self.options['use_progress_bar']):
                pbar = self._tqdm(range(self.options['break_param'] - skip_steps), file=stdout)
            else:
                pbar = range(self.options['break_param'] - skip_steps)
            #Case distinction between sym <-> asym and indices <-> no indices
            if self.options['batch_size'] > 0:
                indices = np.random.randint(low=0, high=self._objective.batch_size, size=(self.options['break_param'] - skip_steps, self.options['batch_size']))
                costs = np.zeros(self.options['break_param'] - skip_steps)
                for i in pbar:
                    temp_cpv, costs[i] = self._optimise_step(temp_cpv, n_jobs, step=i + 1, indices = indices[i])
                    res.add_step(temp_cpv.copy())
            else:
                costs = np.zeros(self.options['break_param'] - skip_steps)
                for i in pbar:
                    temp_cpv, costs[i] = self._optimise_step(temp_cpv, n_jobs, step=i + 1)
                    res.add_step(temp_cpv.copy())
            #Plot Optimisation run, if wanted (only possible for asymmetric gradient, since only there the cost function is calculated without further effort)
            if(self.options['plot_run']):
                assert not self.options['symmetric_gradient'], 'Plotting only supported for asymmetric numerical gradient.'
                plt.plot(range(self._break_param), costs)
                plt.yscale('log')
                plt.savefig(dir_path + '/../../plots/GD_Optimisation.png')
            if(self.options['use_progress_bar']):
                pbar.close()
        return res

#    def configure_fig(fig):
#        fig.yscale('log')
#        fig.title('Bla')
#    
#    options = {'func': configure_fig}
    
    @abc.abstractmethod
    def _optimise_step(self, temp_cpv: np.ndarray, _n_jobs: Integral, step: Integral):
        """
        Perform Optimiser specific update rule and return new parameters
        
        Parameters
        ----------
        temp_cpv: np.ndarray        current parameters
        _n_jobs: Integral        Number of jobs to rum parallel
        step: Integral        number of current step
        """
        raise NotImplementedError()
    
    ###############################################################
    #                                                             #
    #                                                             #
    #                    Symmetric, no indices                    #
    #                                                             #
    #                                                             #
    ###############################################################
    
    def _get_gradients_sym(self, temp_cpv, _n_jobs):
        joined_dict = {**{str(self._circuit_param[i]): temp_cpv[i] for i in range(self._n_param)}}
        # backend options: -'loky'               seems to be always the slowest
        #                   'multiprocessing'   crashes where both other options do not
        #                   'threading'         supposedly best option, still so far slower than
        #                                       seqquential _get_gradients()
        # 1. Get single energies via joblib.Parallel
        # Format E_1 +eps, E_1 - eps
        # Potential issue: Need 2*n_param copies of joined_dict
        _costs = joblib.Parallel(n_jobs=_n_jobs, backend="loky")(
            joblib.delayed(self._get_single_cost_sym)(joined_dict.copy(), j)
            for j in range(2 * self._n_param)
        )
        # 2. Return gradients
        # Make np array?
        _costs = np.array(_costs).reshape((self._n_param, 2))
        gradients_values = np.matmul(_costs, np.array((1, -1))) / (2 * self.options['eps'])
        return gradients_values, 0.5 * (_costs[0][0] + _costs[0][1])

    def _get_single_cost_sym(self, joined_dict, j):
        # Simulate wavefunction at p_j +/- eps
        # j even: +  j odd: -

        # Alternative: _str_cp = str(self.circuit_param[np.divmod(j,2)[0]])
        joined_dict[str(self._circuit_param[np.divmod(j, 2)[0]])] = (
            joined_dict[str(self._circuit_param[np.divmod(j, 2)[0]])]
            + 2 * (0.5 - np.mod(j, 2)) * self.options['eps']
        )
        wf = self._objective.simulate(param_resolver=cirq.ParamResolver(joined_dict))
        return self._objective.evaluate(wf)
    
    ###############################################################
    #                                                             #
    #                                                             #
    #                    Symmetric, with indices                  #
    #                                                             #
    #                                                             #
    ###############################################################
    
    def _get_gradients_sym_indices(self, temp_cpv, _n_jobs, indices: Optional[List[int]] = None):
        joined_dict = {**{str(self._circuit_param[i]): temp_cpv[i] for i in range(self._n_param)}}
        _costs = joblib.Parallel(n_jobs=_n_jobs, backend="loky")(
            joblib.delayed(self._get_single_cost_sym_indices)(joined_dict.copy(), j, indices)
            for j in range(2 * self._n_param)
        )
        _costs = np.array(_costs).reshape((self._n_param, 2))
        gradients_values = np.matmul(_costs, np.array((1, -1))) / (2 * self.options['eps'])
        return gradients_values, 0.5 * (_costs[0][0] + _costs[0][1])
    
    def _get_single_cost_sym_indices(self, joined_dict, j, indices: Optional[List[int]] = None):
        joined_dict[str(self._circuit_param[np.divmod(j, 2)[0]])] = (
            joined_dict[str(self._circuit_param[np.divmod(j, 2)[0]])]
            + 2 * (0.5 - np.mod(j, 2)) * self.options['eps']
        )
        wf = np.zeros(shape=(len(indices), self._objective._N), dtype=np.complex64)
        for k in range(len(indices)):
            wf[k] = self._objective.simulate(param_resolver=cirq.ParamResolver(joined_dict), initial_state=self._objective._initial_wavefunctions[indices[k]])
        return self._objective.evaluate(wf, options={'indices': indices})
    
    ###############################################################
    #                                                             #
    #                                                             #
    #                    Asymmetric, no indices                   #
    #                                                             #
    #                                                             #
    ###############################################################
    
    def _get_gradients_asym(self, temp_cpv, _n_jobs):
        joined_dict = {**{str(self._circuit_param[i]): temp_cpv[i] for i in range(self._n_param)}}
        
        _costs = joblib.Parallel(n_jobs=_n_jobs, backend="loky")(
            joblib.delayed(self._get_single_cost_asym)(joined_dict.copy(), j)
            for j in range(self._n_param + 1)
        )
        _costs = np.array(_costs)
        gradients_values = (_costs[1:] - _costs[0]) / (self.options['eps'])
        return gradients_values, _costs[0]

    def _get_single_cost_asym(self, joined_dict, j):
        if(j>0):
            joined_dict[str(self._circuit_param[j-1])] = (
                joined_dict[str(self._circuit_param[j-1])] + self.options['eps']
            )
        wf = self._objective.simulate(param_resolver=cirq.ParamResolver(joined_dict))
        return self._objective.evaluate(wf)
    
    ###############################################################
    #                                                             #
    #                                                             #
    #                    Asymmetric, with indices                 #
    #                                                             #
    #                                                             #
    ###############################################################

    def _get_gradients_asym_indices(self, temp_cpv, _n_jobs, indices: Optional[List[int]] = None):
        joined_dict = {**{str(self._circuit_param[i]): temp_cpv[i] for i in range(self._n_param)}}
        
        _costs = joblib.Parallel(n_jobs=_n_jobs, backend="loky")(
            joblib.delayed(self._get_single_cost_asym_indices)(joined_dict.copy(), j, indices)
            for j in range(self._n_param + 1)
        )
        _costs = np.array(_costs)
        gradients_values = (_costs[1:] - _costs[0]) / (self.options['eps'])
        return gradients_values, _costs[0]

    def _get_single_cost_asym_indices(self, joined_dict, j, indices: Optional[List[int]] = None):
        if(j>0):
            joined_dict[str(self._circuit_param[j-1])] = (
                joined_dict[str(self._circuit_param[j-1])] + self.options['eps']
            )
        wf = np.zeros(shape=(len(indices), self._objective._N), dtype=np.complex64)
        for k in range(len(indices)):
            wf[k][:] = self._objective.simulate(param_resolver=cirq.ParamResolver(joined_dict), initial_state=self._objective._initial_wavefunctions[indices[k]])
        return self._objective.evaluate(wf, options={'indices': indices})
    
    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                'options': self.options,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])
