from __future__ import annotations

from collections.abc import Callable
import cirq
import importlib
from numbers import Real
import numpy as np
from scipy.integrate import quad
from typing import  Dict, List, Tuple, Union 

from fauvqe.models.spinmodel_fc import SpinModelFC
from fauvqe.models.spinmodel import SpinModel

class Adiabatic(SpinModelFC):
    """
        Adiabatic Sweep through fully connected Spin Models inherits SpinModelFC (-> is itself a SpinModelFC)

        Parameters
        ----------
        H0: initial Hamiltonian (t=0)
        H1: final Hamiltonian (t=T)
        sweep: Optional switch function for adiabatic sweep
        t: Simulation time
        T: total time of adiabatic sweep

        Methods
        ----------
        get_interactions(self) -> List[list]
            Returns
            ---------
            List[list]:
                List of interaction coefficients handed over to constructor
        
        set_Uts(self, trotter_steps: int = 0) -> None
            Returns
            ---------
            void
        
        get_minimal_energy_gap(self, times: np.ndarray = None) -> float
            Returns
            ---------
            float
                Minimum Energy gap along sweep
        
        get_groundstate_at_time(self, time: float) -> np.array
            Returns
            ---------
            np.array
                Ground state of Hamiltonian at specified time
        
        _set_initial_state_for_sweep(self) -> None
            Returns
            ---------
            void
        
        _set_output_state_for_sweep(self) -> None
            Returns
            ---------
            void
    """
    def __init__(self, 
                 H0: Union[SpinModelFC, SpinModel],
                 H1: Union[SpinModelFC, SpinModel],
                 sweep: Callable = None,
                 t: Real = 0,
                 T: Real = 1):
        assert H0.qubittype == H1.qubittype, "Qubit types incompatible, received \nH0: {} \nand \nH1: {}".format(H0.qubittype, H1.qubittype)
        
        assert (H0.n == H1.n).all(), "Qubit numbers incompatible, received \nH0: {} \nand \nH1: {}".format(H0.n, H1.n)
        
        assert t >= 0 and t <= T, "Simulation time incompatible with adiabatic sweep time, received \nt= {} \nand \nT= {}".format(t, T)
        
        if(sweep is None):
            sweep = lambda time: time/T
        else:
            assert abs(sweep(0))<1e-13 and abs(sweep(T) - 1)<1e-13, "Handed sweep is not a switch function, instead sweep(0)= {} and sweep(T) = {}".format(sweep(0), sweep(T))
        
        self._H0 = H0
        self._H1 = H1
        
        if(isinstance(H0, SpinModel)):
        if(isinstance(H0, SpinModel)):
            #self._H0 = SpinModelFC.toFC(H0)
            self._H0.j = SpinModelFC.toFC(H0)
        
        if(isinstance(H1, SpinModel)):
        if(isinstance(H1, SpinModel)):
            #self._H1 = SpinModelFC.toFC(H1)
            self._H1.j = SpinModelFC.toFC(H1)
        
        self.t = t
        self.T = T
        self._sweep = sweep
        
        j_tot, h_tot, TwoQubitGates, SingleQubitGates = self.get_interactions()
        
        super().__init__(
                H0.qubittype, 
                np.array(H0.n),
                np.array(j_tot),
                np.array(h_tot),
                TwoQubitGates,
                SingleQubitGates,
                t
        )
        self.min_gap = None
        self.initial = None
        self.output = None
        self._default_time_grid = np.linspace(0, self.T, self.T + 1)
    
    def get_interactions(self) -> List[list]:
        self.energy_fields = [*self._H0.energy_fields, *self._H1.energy_fields]
        
        l = self._sweep(self.t)
        
        j0 = np.transpose(self._H0.j, (4, 0, 1, 2, 3))
        j1 = np.transpose(self._H1.j, (4, 0, 1, 2, 3))
        h0 = np.transpose(self._H0.h, (2, 0, 1))
        h1 = np.transpose(self._H1.h, (2, 0, 1))
        
        j_tot = np.array([*(1-l)*j0, *l*j1])
        h_tot = np.array([*(1-l)*h0, *l*h1])
        
        TwoQubitGates = [*self._H0._TwoQubitGates, *self._H1._TwoQubitGates]
        SingleQubitGates = [*self._H0._SingleQubitGates, *self._H1._SingleQubitGates]
        
        return j_tot, h_tot, TwoQubitGates, SingleQubitGates
    
    def copy(self) -> Adiabatic:
        self_copy = Adiabatic( 
                self._H0,
                self._H1,
                self._sweep,
                self.t,
                self.T 
        )
        
        self_copy.circuit = self.circuit.copy()
        self_copy.circuit_param = self.circuit_param.copy()
        self_copy.circuit_param_values = self.circuit_param_values.copy()
        self_copy._hamiltonian = self._hamiltonian.copy()
        
        if self.eig_val is not None: self_copy.eig_val = self.eig_val.copy()
        if self.eig_vec is not None: self_copy.eig_vec = self.eig_vec.copy()
        if self._Ut is not None: self_copy._Ut = self._Ut.copy()

        return self_copy
    
    #Overrides SpinModelFC's function
    def _set_hamiltonian(self, reset: bool = True) -> None:
        """
        Append or Reset Hamiltonian; Combine Hamiltonians:
            (1 - sweep(t)) * H0 + sweep(t) * H1
        
        Parameters
        ----------
        self
        reset: bool, indicates whether to reset or append Hamiltonian
        
        Returns
        -------
        void 
        """
        if reset:
            self._hamiltonian = cirq.PauliSum()
        self._hamiltonian = (1-self._sweep(self.t)) * self._H0._hamiltonian + self._sweep(self.t) * self._H1._hamiltonian
    
    #Only Trotterization with adiabatic assumption possible -> alternative Integrate Schrödinger equation directly
    def set_Uts(self, trotter_steps: int = 0) -> None:
        """
        Set sequence of time evolution unitaries to perform sweep, Time evolution generated by time-dependent Hamiltonian is approximated by adiabatic approximation:
        
        Parameters
        ----------
        self
        trotter_steps: int, Finesse of adiabatic approximaiton
        
        Returns
        -------
        void 
        """
        #TODO rename all self._Ut to self._Uts?
        if(trotter_steps == 0):
            trotter_steps = int(self.T)
        delta_t = self.T / trotter_steps
        
        self._Uts = []
        for m in range(trotter_steps):
            hamiltonian = ((1 - self._sweep(m*delta_t)) * self._H0._hamiltonian + self._sweep(m*delta_t) * self._H1._hamiltonian)
            #print(hamiltonian)
            hamiltonian = hamiltonian.matrix()
            eig_val, eig_vec =  np.linalg.eigh(hamiltonian)
            self._Uts.append( 
                np.matmul(np.matmul(eig_vec, np.diag( np.exp( -1j * delta_t * eig_val ) ), dtype = np.complex64), eig_vec.conjugate().transpose())
            )
        #returns into self._Uts the Trotter factors to further use and single time execution
    
    def get_minimal_energy_gap(self, times: np.ndarray = None, reset: bool = False) -> float:
        """
        Calculates the energy gaps in a scan through the adiabatic sweep, saves them to self.gaps and returns the minimum:
        
        Parameters
        ----------
        self
        times: np.ndarray, Array of times the gap should be scanned through
        
        Returns
        -------
        float
            Minimum Energy Gap
        """
        if self.min_gap is not None and not reset:
            return self.min_gap
        if times is None:
            times = self._default_time_grid
        #print(times)
        gaps = []
        groundstates = []
        first_excited = []
        for t in times:
            self.t = t
            self._set_hamiltonian()
            self.diagonalise(solver="numpy")
            gaps.append(self.eig_val[1] - self.eig_val[0])
            groundstates.append(self.eig_vec.transpose()[0])
            first_excited.append(self.eig_vec.transpose()[1])
            if t == 0:
                self.initial = self.eig_vec.transpose()[0]
            if t == self.T:
                self.output = self.eig_vec.transpose()[0]
                self.gs_energy = self.eig_val[0]
        self.min_gap = min(gaps)
        self.min_gap_t = int(np.where(gaps == self.min_gap)[0][0])
        self.gaps = gaps
        self.groundstates = groundstates
        self.first_excited = first_excited
        return self.min_gap

    def get_groundstate_at_time(self, time: float) -> np.array:
        """
        Calculate the ground state of the Hamiltonian H(t) at specified time t:
        
        Parameters
        ----------
        self
        time: float, time t
        
        Returns
        -------
        np.array
            ground state of H(t)
        """
        if(self.t != time):
            self.t = time
            self._set_hamiltonian()
            self.diagonalise(solver="numpy")
        elif(self.eig_vec is None):
            self.diagonalise(solver="numpy")
        return self.eig_vec.transpose()[0]
    
    def _set_initial_state_for_sweep(self) -> None:
        """
        Sets initial state for sweep which is ground state of H0 by calling get_groundstate_at_time:
        
        Parameters
        ----------
        self
        
        Returns
        -------
        void 
        """
        self.initial = self.get_groundstate_at_time(0)
        
    def _set_output_state_for_sweep(self) -> None:
        """
        Sets targeted output state of sweep which is ground state of H1 by calling get_groundstate_at_time:
        
        Parameters
        ----------
        self
        
        Returns
        -------
        void 
        """
        self.output = self.get_groundstate_at_time(self.T)

    def energy(self) -> Tuple[np.ndarray, np.ndarray]:
        return [*((1 - self._sweep(self.t)) * np.array( self._H0.energy())),
                  *(self._sweep(self.t) * np.array(self._H1.energy()))]

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "H0": self._H0,
                "H1": self._H1,
                "sweep": self._sweep,
                "t": self.t,
                "T": self.T
            },
            "params": {
                "circuit": self.circuit,
                "circuit_param": self.circuit_param,
                "circuit_param_values": self.circuit_param_values,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        inst = cls(**dct["constructor_params"])

        inst.circuit = dct["params"]["circuit"]
        inst.circuit_param = dct["params"]["circuit_param"]
        inst.circuit_param_values = dct["params"]["circuit_param_values"]

        return inst