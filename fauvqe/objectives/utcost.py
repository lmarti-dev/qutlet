"""
    Implementation of the Frobenius distance between a given approximate time evolution and the exact time evolution of the system hamiltonian as objective function for an AbstractModel object.
"""
import cirq
import numpy as np

from importlib import import_module
from joblib import delayed, Parallel
from multiprocessing import cpu_count
from numbers import Integral, Real
from typing import Literal, Dict, Optional, List

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel

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
                "m"         -> np.uint
                    Trotter number (Exact if 0 or negative)
                "q"         -> np.uint
                    Trotter-Suzuki order (Exact if 0 or negative)

                "initial_wavefunctions"  -> np.ndarray      if None Use exact UtCost, otherwise batch wavefunctions for random batch cost. Also overwrites evaluate(), the NotImplementedError should never be raised.
                
                "time_steps"  -> List[Int]      List of numbers of circuit runs to be considered in UtCost in ascending order
                
                "use_progress_bar"  -> bool      Determines whether to use a progress bar while initialising the batch wavefunctions (python module tqdm required)

    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <UtCost field=self.field>
    """
    trotter  = import_module("fauvqe.circuits.trotter")

    def __init__(   self,
                    model: AbstractModel, 
                    t: Real, 
                    m: np.uint = 0,
                    q: np.uint = 1,
                    initial_wavefunctions: Optional[np.ndarray] = None,
                    time_steps: List[int] = [1],
                    use_progress_bar: bool = False,
                    dtype: np.dtype = np.complex128):
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
        self._m = m
        self._q = q
        self._use_progress_bar = use_progress_bar
        self._N = 2**np.size(model.qubits)
        self._time_steps = time_steps
        self._dtype = dtype
        
        #Set appropiate evaluate function either state or unitary based
        if (initial_wavefunctions is None):
            self.batch_size = 0
            self.evaluate = self.evaluate_op
        else:
            assert(np.size(initial_wavefunctions[0,:]) == 2**np.size(model.qubits)),\
                "Dimension of given batch_wavefunctions do not fit to provided model; n from wf: {}, n qubits: {}".\
                    format(np.log2(np.size(initial_wavefunctions[0,:])), np.size(model.qubits))
            self.batch_size = np.size(initial_wavefunctions[:,0])
            #self._init_batch_wfcts()
            self.evaluate = self.evaluate_batch

        #Init Ut
        if self._m == 0 or self._q == 0:
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
            self._Ut_error = 0
            if initial_wavefunctions is not None:  self._output_wavefunctions = self.get_output_wavefunctions()
        else:
            self.trotter_circuit = self.get_trotter_circuit()
            #assert initial_wavefunctions is not None, 'Please provide batch wavefunctions for Trotter Approximation'
            if initial_wavefunctions is None:
                self._Ut = cirq.unitary(self.trotter_circuit)
                self._Ut_error = self.get_Ut_error()
            else:
                self._output_wavefunctions = self.get_output_wavefunctions()
                self._Ut_error = self.get_Ut_error()


        

    def get_trotter_circuit(  self,
                                options: Dict = {}):
        """
            This function initialises the trotter circuit.

            Make sure that hamiltonian is potentially a time-dependent function

            Also potentially allow for t0 != 0
        """
        self.trotter.options = {    "append": False,
                                    "return": True,
                                    "hamiltonian": self.model.hamiltonian,
                                    "trotter_number" : self._m,
                                    "trotter_order" : self._q,
                                    "t0": 0, 
                                    "tf": float(self.t)}
        self.trotter.options.update(options)
        return self.trotter.set_circuit(self)
    
    def get_output_wavefunctions(self):
        """
        This function initialises the initial and output batch wavefunctions as sampling data and sets self._output_wavefunctions.
        
        Parameters
        ----------
        self
        
        Returns
        ---------
        void
        """
        _output_wavefunctions = np.empty(shape=( len(self._time_steps), *self._initial_wavefunctions.shape), dtype=self._dtype)
        if(self._m < 1):
            for step in range(len(self._time_steps)):
                _output_wavefunctions[step] = (np.linalg.matrix_power(self._Ut, self._time_steps[step]) @ self._initial_wavefunctions.T).T
        else:
            pbar = self.create_range(self.batch_size, self._use_progress_bar)
            #Didn't find any cirq function which accepts a batch of initials
            # TODO: replace mul*self.trotter_circuit with get_trotter_circuit() to allow for time dependent hamiltonian 
            for step in range(len(self._time_steps)):
                if(step != 0):
                    ini = _output_wavefunctions[step - 1]
                    mul = self._time_steps[step] - self._time_steps[step - 1]
                else:
                    ini = self._initial_wavefunctions
                    mul = self._time_steps[step]
                #Use multiprocessing
                tmp = Parallel(n_jobs=min(cpu_count(), self.batch_size))(
                delayed(self.model.simulator.simulate)( mul * self.trotter_circuit, initial_state=ini[k]) for k in pbar
                )
                for k in range(self._initial_wavefunctions.shape[0]):
                    _output_wavefunctions[step][k] = tmp[k].state_vector() / np.linalg.norm(tmp[k].state_vector())
            if(self._use_progress_bar):
                pbar.close()
        return _output_wavefunctions

    def get_Ut_error(self):
        """
            This function approximates the error of the ground truth Ut 
            by comparing it to the trotter_numer self._m +1 circuit on the batch wavefunctions


        """
        # this does not (yet) work for multiple timesteps
        if len(self._time_steps) > 1:
            return -1

        self.trotter.options.update({"trotter_number" : self._m+1})
        mp1_trotter_circuit = self.trotter.set_circuit(self)
        self.trotter.options.update({"trotter_number" : self._m})

        if self.batch_size == 0:
            _mp1_final_states = cirq.unitary(mp1_trotter_circuit)
        else:
            tmp = Parallel(n_jobs=min(cpu_count(), self.batch_size))(
                        delayed(self.model.simulator.simulate)( mp1_trotter_circuit, 
                                                            initial_state=self._initial_wavefunctions[k]) for k in range(self.batch_size)
                        )
            _mp1_final_states = []

            for k in range(self._initial_wavefunctions.shape[0]):
                _mp1_final_states.append(tmp[k].state_vector() / np.linalg.norm(tmp[k].state_vector()))

        return self.evaluate(_mp1_final_states)

    def evaluate(self):
        raise NotImplementedError
            
    def evaluate_op(self, wavefunction: np.ndarray) -> np.float64:
        cost = 0

        # This assumes that wavefunction is a unitary
        # There is the situation where one wants the unitary cost of batched wavefunctions
        # Then one needs to produce the circuit unitary as wavefunction
        assert np.shape(wavefunction) == (self._N, self._N)

        for step in range(len(self._time_steps)):
            cost += 1 - abs(np.trace( 
                np.linalg.matrix_power( np.matrix.getH(self._Ut), self._time_steps[step]) @ \
                np.linalg.matrix_power( wavefunction, self._time_steps[step]))) / self._N
        return 1 / len(self._time_steps) * cost
    
    def evaluate_batch(self, wavefunction: np.ndarray, options: dict = {'indices': None}) -> np.float64:
        #assert (options['indices'] is not None), 'Please provide indices for batch'
        #assert wavefunction.size == (len(self._time_steps) * len(options['indices']) * self._N), 'Please provide one wavefunction for each time step. Expected: {} Received: {}'.\
        #    format( len(self._time_steps), wavefunction.size / self._N )
        cost = 0

        if options.get('indices') is None:
            options['indices'] = range(self.batch_size)

        #TODO This might be wrong
        for step in range(len(self._time_steps)):
            cost += np.sum(1 - abs(np.sum(np.conjugate(wavefunction[step])*
                                            self._output_wavefunctions[step][options.get('indices')], axis=1)))
        return 1 / len(self._time_steps) * 1/len(options.get('indices')) * cost

    #Need to overwrite simulate from parent class in order to work
    def simulate(self, param_resolver, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        #TODO Rewrite this!
        #return unitary if self.batch_size == 0
        if self.batch_size == 0:
            return cirq.resolve_parameters(self._model.circuit, param_resolver).unitary()
        else:
            if initial_state is None:
                #Set all initial batch states as initial states
                initial_state = self._initial_wavefunctions
            
            output_state = np.empty(shape=(len(self._time_steps), *initial_state.shape), dtype = self._dtype)
            #print(np.shape(output_state))
            
            if len(initial_state) == self._N:
                ini = initial_state
            else:
                ini = initial_state[0]

            mul = self._time_steps[0]
            wf = self._model.simulator.simulate(
                    mul * self._model.circuit,
                    param_resolver=param_resolver,
                    initial_state=ini,
                    ).state_vector()
            output_state[0] = wf/np.linalg.norm(wf)

            #Run through all states given in initial state
            for i_state in range(1,round(len(initial_state)/self._N)):
                wf = self._model.simulator.simulate(
                    mul * self._model.circuit,
                    param_resolver=param_resolver,
                    initial_state=initial_state[i_state],
                    ).state_vector()
                output_state[0, i_state] = wf/np.linalg.norm(wf)

            #Calculate output_state for more time steps if len(self._time_steps > 0)
            #TODO adapt this in case both time_steps and batch wavefunctions are used
            for step in range(1, len(self._time_steps)):
                ini = output_state[step-1]
                mul = self._time_steps[step] - self._time_steps[step-1]
                wf = self._model.simulator.simulate(
                    mul * self._model.circuit,
                    param_resolver=param_resolver,
                    initial_state=ini,
                    ).state_vector()
                output_state[step] = wf/np.linalg.norm(wf)

            # remove not used extra dimensions
            #return np.squeeze(output_state)
            return np.squeeze(output_state)
            
    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "t": self.t, 
                "m": self._m,
                "initial_wavefunctions": self._initial_wavefunctions
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<UtCost t={}>".format(self.t)