"""
    Implementation of the Frobenius distance between a given approximate time evolution and the exact time evolution of the system hamiltonian as objective function for an AbstractModel object.
"""
from typing import Literal, Dict, Optional, List
from numbers import Integral, Real

import numpy as np
from tqdm import tqdm

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel
#from fauvqe import Objective, AbstractModel
import cirq

class UtCost(Objective):
    """
    U(t) cost objective

    This class implements as objective the difference between exact U(t) and VQE U(t)
    of the linked model.

    Parameters
    ----------
    model: AbstractModel, The linked model
    options:    "t"         -> Float
                    t in U(t)
                "order"         -> np.uint
                    Trotter approximation order (Exact if 0 or negative)
                "initial_wavefunctions"  -> np.ndarray      if None Use U csot, otherwise batch wavefunctions for random batch cost
                "sample_size"  -> Int      < 0 -> state vector, > 0 -> number of samples

    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <UtCost field=self.field>
    """
    def __init__(   self,
                    model: AbstractModel, 
                    t: Real, 
                    order: np.uint = 0,
                    initial_wavefunctions: Optional[np.ndarray] = None,
                    use_progress_bar: bool = False):
        #Idea of using variable "method" her instead of boolean is that 
        #This allows for more than 2 Calculation methods like:
        #   Cost via exact unitary
        #   Cost via Trotter unitary
        #       -Cost via random batch sampling of these, but exact state vector
        #       -Cost via random batch sampling of these & sampling state
        #
        #   Based on the method, different further parameters are needed hence rather use dict
        # To be implemented: U exact unitatry cost, U exact random batch sampling cost with wf 
        
        #Make sure correct Ut is used
        self.t = t
        super().__init__(model)
        
        self._initial_wavefunctions = initial_wavefunctions
        self._order = order
        self._use_progress_bar = use_progress_bar
        self._N = 2**np.size(model.qubits)
        
        if self._order == 0:
            if t != model.t:
                model.t = t
                model.set_Ut()
                self._Ut = model._Ut.view()
            else:
                try:
                    #Fails if does not exist
                    self._Ut = model._Ut.view()
                except:
                    model.set_Ut()
                    self._Ut = model._Ut.view()
        else:
            assert initial_wavefunctions is not None, 'Please provide batch wavefunctions for Trotter Approximation'
            self._init_trotter_circuit()
        if (initial_wavefunctions is None):
            self.batch_size = 0
        else:
            assert(np.size(initial_wavefunctions[0,:]) == 2**np.size(model.qubits)),\
                "Dimension of given batch_wavefunctions do not fit to provided model; n from wf: {}, n qubits: {}".\
                    format(np.log2(np.size(initial_wavefunctions[0,:])), np.size(model.qubits))
            self.batch_size = np.size(initial_wavefunctions[:,0])
            self._init_batch_wfcts()

    def _init_trotter_circuit(self):
        """
        This function initialises the circuit for Trotter approximation and sets self.trotter_circuit
        
        Parameters
        ----------
        self
        
        Returns
        ---------
        void
        """
        self.trotter_circuit = cirq.Circuit()
        hamiltonian = self.model.hamiltonian
        #Loop through all the addends in the PauliSum Hamiltonian
        for pstr in hamiltonian._linear_dict:
            #temp encodes each of the PauliStrings in the PauliSum hamiltonian which need to be turned into gates
            temp = 1
            #Loop through Paulis in the PauliString (pauli[1] encodes the cirq gate and pauli[0] encodes the qubit on which the gate acts)
            for pauli in pstr:
                temp = temp * pauli[1](pauli[0])
            #Append the PauliString gate in temp to the power of the time step * coefficient of said PauliString. The coefficient needs to be multiplied by a correction factor of 2/pi in order for the PowerGate to represent a Pauli exponential.
            self.trotter_circuit.append(temp**np.real(2/np.pi * self.t * hamiltonian._linear_dict[pstr] / self._order))
        #Copy the Trotter layer *order times.
        #self.trotter_circuit = qsimcirq.QSimCircuit(self._order * self.trotter_circuit)
        self.trotter_circuit = self._order * self.trotter_circuit
    
    def _init_batch_wfcts(self):
        """
        This function initialises the initial and output batch wavefunctions as sampling data and sets self._output_wavefunctions.
        
        Parameters
        ----------
        self
        
        Returns
        ---------
        void
        """
        if(self._order < 1):
            self._output_wavefunctions = (self._Ut @ self._initial_wavefunctions.T).T
        else:
            if(self._use_progress_bar):
                pbar = tqdm(range(self._initial_wavefunctions.shape[0]))
            else:
                pbar = range(self._initial_wavefunctions.shape[0])
            self._output_wavefunctions = np.zeros(shape=self._initial_wavefunctions.shape, dtype=np.complex128)
            #Didn't find any cirq function which accepts a batch of initials
            for k in pbar:
                self._output_wavefunctions[k] = self.model.simulator.simulate(
                    self.trotter_circuit,
                    initial_state=self._initial_wavefunctions[k]
                    #dtype=np.complex128
                ).state_vector()
            if(self._use_progress_bar):
                pbar.close()
            #self.trotter_circuit = qsimcirq.QSimCircuit(self.trotter_circuit)
            #start = time()
            #for k in range(self._initial_wavefunctions.shape[0]):
            #    self._output_wavefunctions[k] = self.trotter_circuit.final_state_vector(
            #        initial_state=self._initial_wavefunctions[k],
            #        dtype=np.complex64
            #    )
            #end = time()
            #print(end-start)
    
    def evaluate(self, wavefunction: np.ndarray, options: dict = {'indices': None}) -> np.float64:
        # Here we already have the correct model._Ut
        if self.batch_size == 0:
            #Calculation via Forbenius norm
            #Then the "wavefunction" is the U(t) via VQE
            return 1 - abs(np.trace(np.matrix.getH(self._Ut) @ wavefunction)) / self._N
        else:
            assert (options['indices'] is not None), 'Please provide indices for batch'
            return 1/len(options['indices']) * np.sum(1 - abs(np.sum(np.conjugate(wavefunction)*self._output_wavefunctions[options['indices']], axis=1)))

    #Need to overwrite simulate from parent class in order to work
    def simulate(self, param_resolver, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        #return unitary if self.batch_size == 0
        if self.batch_size == 0:
            return cirq.resolve_parameters(self._model.circuit, param_resolver).unitary()
        else:
            return super().simulate(param_resolver, initial_state)

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "t": self.t, 
                "order": self._order,
                "initial_wavefunctions": self._initial_wavefunctions
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<UtCost t={}>".format(self.t)

    
    
    
'''TEMP:
"""
This abstract class resembles functions and attributes which Gradient Optimiser have in common. It hence inherits from Optimiser().

"""
import abc
import cirq

from fauvqe.objectives.objective import Objective
from fauvqe.optimisers.optimisation_result import OptimisationResult
from fauvqe.optimisers.optimiser import Optimiser

import joblib

import math
import matplotlib.pyplot as plt
import multiprocessing
from numbers import Real, Integral
import numpy as np
import os
dir_path = os.path.abspath(os.path.dirname(__file__))
from sys import stdout
from tqdm import tqdm # progress bar
from typing import Literal, Optional, Dict, List


class GradientOptimiser(Optimiser):
    """Abstract GradientOptimiser class"""

    def __init__(
        self,
        eps: Real = 1e-3,
        eta: Real = 1e-2,
        break_cond: Literal["iterations"] = "iterations",
        break_param: Integral = 100,
        batch_size: Integral = 0,
        optimiser_options: dict = {}
    ):
        """GradientOptimiser
        
        Parameters
        ----------
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
        
        optimiser_options: dict
            Dictionary containing additional options to individualise the optimisation routine. Contains:
                symmetric_gradient: bool
                    Specifies whether to use symmetric numerical gradient or asymmetric gradient (faster by ~ factor 2)
                
                plot_run: bool
                    Plot cost development in optimisation run and save to fauvqe/plots
                
                use_progress_bar: bool
                    Determines whether to use tqdm's progress bar when running the optimisation
        """

        super().__init__()
        assert all(
            isinstance(n, Real) and n > 0.0 for n in [eps, eta]
        ), "Parameters must be positive, real numbers"
        assert (
            isinstance(n, Integral) and n > 0 for n in [break_param, batch_size]
        ), "Parameters must be positive integers"
        valid_break_cond = ["iterations"]
        assert (
            break_cond in valid_break_cond
        ), "Invalid break condition, received: '{}', allowed are {}".format(
            break_cond, valid_break_cond
        )

        # The following attributes remain constant for the lifetime of this optimiser
        self._eps: Real = eps
        self._eta: Real = eta
        self._break_cond: Literal["iterations"] = break_cond
        self._break_param: Integral = break_param
        self._batch_size: Integral = batch_size
        self._circuit_param = objective.model.circuit_param
        
        self._optimiser_options = {'symmetric_gradient': True, 'plot_run': False, 'use_progress_bar': False}
        self._optimiser_options.update(optimiser_options)
        if(self._optimiser_options['symmetric_gradient']):
            self._get_gradients = self._get_gradients_sym
            self._get_single_cost = self._get_single_cost_sym
            nbr_processes = 2 * np.size(self._circuit_param)
        else:
            self._get_gradients = self._get_gradients_asym
            self._get_single_cost = self._get_single_cost_asym
            nbr_processes = np.size(self._circuit_param) + 1
        
        # Determine maximal number of threads and reset qsim 't' flag for n_job = -1 (default)
        if optimiser_options['n_jobs'] < 1:
            # max(n_jobs) = 2*n_params, as otherwise overhead of not used jobs
            optimiser_options['n_jobs'] = int(
                min(
                    max(multiprocessing.cpu_count() / 2, 1),
                    nbr_processes,
                )
            )
            assert optimiser_options['n_jobs'] != 0, "{} {}".format(
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
        self._optimiser_options.update({'n_jobs': n_jobs})
        
        # The following attributes change for each objective
        self._objective: Optional[Objective] = None
        self._n_param: Integral = 0

    def optimise(
        self,
        objective: Objective,
        continue_at: Optional[OptimisationResult] = None,
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

        if self._batch_size > 0:
            indices = np.random.randint(low=0, high=self._objective.batch_size, size=(self._break_param - skip_steps, self._batch_size))
        else:
            indices = [None for k in range(self._break_param - skip_steps)]
        
        #Set progress bar, if wanted
        if(self._optimiser_options['use_progress_bar']):
            pbar = tqdm(range(self._break_param - skip_steps), file=stdout)
        else:
            pbar = range(self._break_param - skip_steps)
        
        costs = [None for k in range(self._break_param - skip_steps)]
        # Do step until break condition
        if self._break_cond == "iterations":
            for i in pbar:
                temp_cpv, costs[i] = self._cpv_update(temp_cpv, n_jobs, step=i + 1, indices = indices[i])
                res.add_step(temp_cpv.copy(), objective = costs[i])
        if(self._optimiser_options['use_progress_bar']):
            pbar.close()
        if(self._optimiser_options['plot_run']):
            plt.plot(range(self._break_param), costs)
            plt.yscale('log')
            plt.savefig(dir_path + '/../../plots/GD_Optimisation.png')
        return res

    @abc.abstractmethod
    def _cpv_update(self, temp_cpv: np.ndarray, _n_jobs: Integral, step: Integral, indices: Optional[List[int]] = None):
        """
        Perform Optimiser specific update rule and return new parameters
        
        Parameters
        ----------
        temp_cpv: np.ndarray        current parameters
        _n_jobs: Integral        Number of jobs to rum parallel
        step: Integral        number of current step
        """
        raise NotImplementedError()
    
    def _get_gradients_sym(self, temp_cpv, _n_jobs, indices: Optional[List[int]] = None):
        joined_dict = {**{str(self._circuit_param[i]): temp_cpv[i] for i in range(self._n_param)}}
        # backend options: -'loky'               seems to be always the slowest
        #                   'multiprocessing'   crashes where both other options do not
        #                   'threading'         supposedly best option, still so far slower than
        #                                       seqquential _get_gradients()
        # 1. Get single energies via joblib.Parallel
        # Format E_1 +eps, E_1 - eps
        # Potential issue: Need 2*n_param copies of joined_dict
        _costs = joblib.Parallel(n_jobs=_n_jobs, backend="loky")(
            joblib.delayed(self._get_single_cost)(joined_dict.copy(), j, indices)
            for j in range(2 * self._n_param)
        )
        # 2. Return gradients
        # Make np array?
        _costs = np.array(_costs).reshape((self._n_param, 2))
        gradients_values = np.matmul(_costs, np.array((1, -1))) / (2 * self._eps)
        return gradients_values, None

    def _get_single_cost_sym(self, joined_dict, j, indices: Optional[List[int]] = None):
        # Simulate wavefunction at p_j +/- eps
        # j even: +  j odd: -

        # Alternative: _str_cp = str(self.circuit_param[np.divmod(j,2)[0]])
        joined_dict[str(self._circuit_param[np.divmod(j, 2)[0]])] = (
            joined_dict[str(self._circuit_param[np.divmod(j, 2)[0]])]
            + 2 * (0.5 - np.mod(j, 2)) * self._eps
        )
        if (indices is None):
            wf = self._objective.simulate(param_resolver=cirq.ParamResolver(joined_dict))
        else:
            wf = np.zeros(shape=(len(indices), self._objective._N), dtype=np.complex64)
            for k in range(len(indices)):
                wf[k] = self._objective.simulate(param_resolver=cirq.ParamResolver(joined_dict), initial_state=self._objective._initial_wavefunctions[indices[k]])
        return self._objective.evaluate(wf, options={'indices': indices})
    
    def _get_gradients_asym(self, temp_cpv, _n_jobs, indices: Optional[List[int]] = None):
        joined_dict = {**{str(self._circuit_param[i]): temp_cpv[i] for i in range(self._n_param)}}
        
        _costs = joblib.Parallel(n_jobs=_n_jobs, backend="loky")(
            joblib.delayed(self._get_single_cost)(joined_dict.copy(), j, indices)
            for j in range(self._n_param + 1)
        )
        _costs = np.array(_costs)
        gradients_values = (_costs[1:] - _costs[0]) / (self._eps)
        return gradients_values, _costs[0]

    def _get_single_cost_asym(self, joined_dict, j, indices: Optional[List[int]] = None):
        if(j>0):
            joined_dict[str(self._circuit_param[j-1])] = (
                joined_dict[str(self._circuit_param[j-1])] + self._eps
            )
        if (indices is None):
            wf = self._objective.simulate(param_resolver=cirq.ParamResolver(joined_dict))
        else:
            wf = np.zeros(shape=(len(indices), self._objective._N), dtype=np.complex64)
            for k in range(len(indices)):
                wf[k][:] = self._objective.simulate(param_resolver=cirq.ParamResolver(joined_dict), initial_state=self._objective._initial_wavefunctions[indices[k]])
        return self._objective.evaluate(wf, options={'indices': indices})

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "eps": self._eps,
                "eta": self._eta,
                "break_cond": self._break_cond,
                "break_param": self._break_param,
                "batch_size": self._batch_size,
                'optimiser_options': self._optimiser_options,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])


'''