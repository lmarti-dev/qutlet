from __future__ import annotations

from collections.abc import Callable
from typing import Dict, Union, List, Optional
from numbers import Real
from fauvqe.objectives.fidelity import Fidelity
from fauvqe.objectives.abstractexpectationvalue import AbstractExpectationValue

import numpy as np
import cirq

from fauvqe.models.coolingmodel import CoolingModel
from fauvqe.models.adiabatic import Adiabatic
from fauvqe.models.spinModel_fc import SpinModelFC
from fauvqe.models.spinModel import SpinModel
from fauvqe.models.abstractmodel import AbstractModel

from fauvqe.utils import ptrace

class CooledAdiabatic(CoolingModel):
    """
       Cooling assisted Adiabatic Sweep through fully connected Spin Models inherits SpinModelFC (-> is itself a SpinModelFC)

        Parameters
        ----------
        H0: initial Hamiltonian (t=0)
        H1: final Hamiltonian (t=T)
        m_anc: Model used for Fridge qubits
        int_gates: Interaction types between system and fridge
        j_int: Interaction coefficients
        sweep: Optional switch function for adiabatic sweep
        t: Simulation time
        T: total time of adiabatic sweep

        Methods
        ----------
        
    """
    def __init__(self, 
                 H0: Union[SpinModelFC, SpinModel],
                 H1: Union[SpinModelFC, SpinModel],
                 m_anc: AbstractModel,
                 int_gates: List[cirq.PauliSum],
                 j_int: np.array,
                 sweep: Callable = None,
                 t: Real = 0,
                 T: Real = 1
                ):
        """
        H0: initial Hamiltonian (t=0)
        H1: final Hamiltonian (t=T)
        t: Simulation time
        T: total time of adiabatic sweep
        m_anc: model defining the ancilla system
        int_gates: a cirq.PauliSum defining the interaction gates between system and ancillas including interaction constants
        j_int: an array encoding the interaction constants between ancillas and system
        """
        m_sys = Adiabatic(H0,
                          H1,
                          sweep,
                          t,
                          T)
        
        self._Uts: Optional[np.ndarray] = None
        
        super().__init__(m_sys,
                 m_anc,
                 int_gates,
                 j_int,
                 t)
    
    def copy(self) -> CooledAdiabatic:
        self_copy = CooledAdiabatic( 
                self.m_sys._H0,
                self.m_sys._H1,
                self.m_anc,
                self._TwoQubitGates[self.nbr_2Q_sys + self.nbr_2Q_anc:self.nbr_2Q],
                self.j_int,
                self.m_sys._sweep,
                self.m_sys.t,
                self.m_sys.T
        )
        
        self_copy.circuit = self.circuit.copy()
        self_copy.circuit_param = self.circuit_param.copy()
        self_copy.circuit_param_values = self.circuit_param_values.copy()
        self_copy.hamiltonian = self.hamiltonian.copy()
        
        if self.eig_val is not None: self_copy.eig_val = self.eig_val.copy()
        if self.eig_vec is not None: self_copy.eig_vec = self.eig_vec.copy()
        if self._Ut is not None: self_copy._Ut = self._Ut.copy()

        return self_copy
    
    #Overrides CoolingModel's function
    def _set_hamiltonian(self, reset: bool = True) -> None:
        if reset:
            self.hamiltonian = cirq.PauliSum()
        self.hamiltonian += self._get_hamiltonian_at_time(self.t)
    
    def _get_hamiltonian_at_time(self, time: Real) -> np.ndarray:
        """
        Append or Reset Hamiltonian; Combine Hamiltonians:
            (1 - sweep(t)) * H0 + sweep(t) * H1
            + m_anc.hamiltonian
            + interaction term
        
        Parameters
        ----------
        self
        time: Real, time at which Hamiltonian shall be calculated
        
        Returns
        -------
        np.ndarray
            Hamiltonian H(t) at time t
        """
        ham = cirq.PauliSum()
        
        self.m_sys.t = time
        self.m_sys._set_hamiltonian()
        self.m_anc.t = time
        self.m_anc._set_hamiltonian()
        
        sys_rows = self.m_sys.n[0]
        new_qubits = [self.qubits[i][j] for i in range(sys_rows, self.n[0], 1) for j in range(self.n[1])]
        ham = self.m_sys.hamiltonian + self.m_anc.hamiltonian.with_qubits(*new_qubits)
        int_gates = self._TwoQubitGates[self.nbr_2Q_sys + self.nbr_2Q_anc:self.nbr_2Q]
        for g in range(len(int_gates)):
            for i in range(self.m_sys.n[0]):
                for j in range(self.m_sys.n[1]):
                    if(self.j_int[g][i][j] != 0):
                        if self.cooling_type == "NA":
                            ham -= self.j_int[g][i][j] * cirq.PauliSum.from_pauli_strings(int_gates[g](self.qubits[i][j], self.qubits[i+sys_rows][j]))
                        else:
                            ham -= self.j_int[g][i][j] * cirq.PauliSum.from_pauli_strings(int_gates[g](self.qubits[i][j], self.qubits[sys_rows][j]))
        
        return ham
    
    #Only Trotterization with adiabatic assumption possible -> alternative Integrate Schrödinger equation directly
    def set_Uts(self, trotter_steps: int = 0) -> None:
        """
        Sets sequence of time evolution unitaries to perform sweep together with cooling system, Time evolution generated by time-dependent Hamiltonian is approximated by adiabatic approximation:
        
        Parameters
        ----------
        self
        trotter_steps: int, Finesse of adiabatic approximaiton
        
        Returns
        -------
        void
        """
        if(trotter_steps == 0):
            trotter_steps = int(self.m_sys.T)
        delta_t = self.m_sys.T / trotter_steps
        
        self._Uts = []
        for m in range(trotter_steps):
            hamiltonian = self._get_hamiltonian_at_time(m*delta_t)
            #print(hamiltonian)
            hamiltonian = hamiltonian.matrix()
            eig_val, eig_vec =  np.linalg.eigh(hamiltonian)
            self._Uts.append( 
                np.matmul(np.matmul(eig_vec, np.diag( np.exp( -1j * delta_t * eig_val ) ), dtype = np.complex64), eig_vec.conjugate().transpose())
            )
    
    def _get_default_trotter_steps(self, nbr_resets: int) -> int:
        """
        Calculates default trotter_steps for sweep through H(t) to get a trotter number divisible by nbr_resets:
        
        Parameters
        ----------
        self
        nbr_resets: int, Number of desired fridge qubit resets along the sweep
        
        Returns
        ----------
        int
            default trotter steps
        """
        return int( int(self.m_sys.T) - (int(self.m_sys.T) % nbr_resets) ) #+ nbr_resets )

    def perform_sweep(self, nbr_resets: int = None, t_steps: int = None, calc_O: bool = True, calc_E: bool = True) -> List[np.ndarray]:
        """
        Calculate the whole sweep:
            - Configures Uts and nbr_resets
            - Initialize with H0 groundstate
            - Repeatedly time evolve and reset fridge qubits
            - Calculate overlaps and energies on the fly
        
        Parameters
        ----------
        self
        nbr_resets: int, Number of desired fridge qubit resets along the sweep
        calc_O: bool, decides whether overlaps with instantaneous groundstate are calculated
        calc_E: bool, decides whether target Hamiltonian's energies are calculated
        
        Returns
        ----------
        List[np.array]
            final: final state of cooled adiabatic sweep
            fids: Instantaneous groundstate overlaps
            energies: H1 energies
        """
        _n = np.size(self.m_anc.qubits)
        _N = 2**( _n )
        _n_full = np.size(self.qubits)
        _N_sys = 2**(_n_full - _n)
        min_gap = 0
        
        #Set number of resets
        if nbr_resets is None:
            if min_gap == 0:
                if t_steps is None:
                    min_gap = self.m_sys._get_minimal_energy_gap()
                else:
                    min_gap = self.m_sys._get_minimal_energy_gap(
                        np.linspace(0, self.m_sys.T, t_steps+1)
                    )
            dt = 2 * np.pi / min_gap
            nbr_resets = int(self.m_sys.T / dt)
            if(nbr_resets == 0):
                nbr_resets = 1
        self.nbr_resets = nbr_resets
        if(t_steps is None):
            t_steps = self._get_default_trotter_steps(nbr_resets)
        assert t_steps % nbr_resets == 0, "Need to have a multiple of reset number for trotter number, received: {} and {}".format(nbr_resets, t_steps)

        #Set Uts for sweep
        if self._Uts is None or (len(self._Uts) % nbr_resets != 0 ):
            self.set_Uts(t_steps)
        else:
            t_steps = len(self._Uts)
        steps_between_resets = int( t_steps / nbr_resets )
        print("Perform sweep with {} steps, {} resets and {} steps between resets".format(t_steps, nbr_resets, steps_between_resets))
        
        #Instantiate figures of interest if desired
        if(calc_O):
            if(self.m_sys.output is None):
                min_gap = self.m_sys._get_minimal_energy_gap(
                    np.linspace(0, self.m_sys.T, t_steps+1), 
                    reset = True
                )
            fid = Fidelity(self.m_sys, self.m_sys.output)
        fids = []
        if(calc_E):
            energy = AbstractExpectationValue(self.m_sys._H1)
        energies = []
        #Get initial state from groundstate of m_sys.hamiltonian(t=0)
        if(self.m_sys.initial is None):
            self.m_sys._set_initial_state_for_sweep()
        fridge_gs = np.zeros(shape=(_N, _N))
        fridge_gs[0, 0] = 1.0
        initial = np.kron(self.m_sys.initial.reshape(_N_sys, 1) @ self.m_sys.initial.conjugate().reshape(1, _N_sys),
                         fridge_gs)
        # Do the sweep with intermediate resets
        final = initial
        for step in range(t_steps):
            final = self._Uts[step] @ final @ self._Uts[step].transpose().conjugate()
            if((step+1) % steps_between_resets == 0):
                sys_state = ptrace(final, range(_n_full - _n, _n_full, 1))
                if(calc_O):
                    #fid = Fidelity(self.m_sys, self.m_sys.groundstates[step+1])
                    fids.append(fid.evaluate(sys_state)[0][0])
                if(calc_E):
                    energies.append(energy.evaluate(sys_state))
                final =  np.kron( sys_state, fridge_gs)
        return final, fids, energies
    
    def get_theory_bounds(self, epsilon: float = None) -> Dict:
        if epsilon is None:
            epsilon = 1 - np.exp(-0.5* np.pi / self.m_sys.T )
            print(epsilon)
        omega_anc = self.m_anc.eig_val[1] - self.m_anc.eig_val[0]
        omega_sys = self.m_sys.min_gap
        __n = self.m_sys.n[0] * self.m_sys.n[1]
        qubits = [cirq.GridQubit(0, j) for j in range(__n)]
        S = 0
        eig_vec = [
            self.m_sys.groundstates[self.m_sys.min_gap_t], 
            self.m_sys.first_excited[self.m_sys.min_gap_t]
        ]
        for pauli in [cirq.X, cirq.Y, cirq.Z]:
            S += 1/3 * abs(eig_vec[1].transpose() @ (cirq.IdentityGate(__n).on(*qubits) * pauli.on(qubits[0])).matrix(qubits) @ eig_vec[0])
        return {
            'alpha_benchmark': __n * np.sqrt(epsilon*(1-epsilon))/(4*np.pi*epsilon) \
                / self.m_sys.T / self.m_sys.min_gap / S,
            'alpha_high': 2*omega_anc,
            'gap_difference_benchmark': 0,
            'gap_difference_high': omega_anc + omega_sys,
            'dt_between_resets_benchmark': 2*np.pi/omega_anc,
            'dt_between_resets_high': self.m_sys.T
        }


    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "H0": self.m_sys._H0,
                "H1": self.m_sys._H1,
                "m_anc": self.m_anc,
                "int_gates": self._TwoQubitGates[self.nbr_2Q_sys + self.nbr_2Q_anc:self.nbr_2Q],
                "j_int": self.j_int,
                "sweep": self.m_sys._sweep,
                "t": self.m_sys.t,
                "T": self.m_sys.T
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