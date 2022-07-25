from __future__ import annotations

import cirq
import importlib
from numbers import Real
import numpy as np
from typing import Dict, Literal, Optional, Tuple, Union

from fauvqe.models.spinmodel_fc import SpinModelFC

class ANNNI(SpinModelFC):
    """
        This class implements the axial next-nearest neighbour Ising (ANNNI) model based on an fully connected SpinModel type.
        Its Hamiltonian reads:
            H = -(Σ_<i,j > J_ij Z_i Z_j - Σ_<i,j NNN>  κ_ij Z_i Z_j) - Σ_i h_i X_i

        Note that here we set the qubittype to GridQubit as it is our default anyway. 
        Further we want to allow for an constructor where J, k, h are numbers. 
        If both J and k are number require optional boundaries given.

        TODO: possibly extend this to more general coupling or external fields

        Parameters
        ----------
        n:          Union[Integer, np.ndarray]
                    The qubit layout of the lattice
                    If int given n -> [n, 1]

        J:          Union[Real, np.ndarray]
                    The nearest neighbour coupling

        k:          Union[Real, np.ndarray]
                    The next nearest neighbour coupling

        h:          Union[Real, np.ndarray]
                    The external field strength in X-direction

        boundaries: Optional[np.ndarray]
                    boundary conditions of the model calc if J or k are nd.arrays
                    Otherwise require to be provided

        t:          Optional[Number]
                    Final simulation time; likely not used so much when ANNNI is used with DrivenModel

        Methods
        ----------
        TODO Add descriptions here

        __repr__() : str
            Returns
            ---------
            str:
                <ANNNI J, k, h, boundaries>

            Typ hinting Functions:
            https://stackoverflow.com/questions/37835179/how-can-i-specify-the-function-type-in-my-type-hints
    """
    def __init__(   self,
                    n: Union[int, np.ndarray], 
                    J: Union[Real, np.ndarray, Tuple[np.ndarray,np.ndarray]], 
                    k: Union[Real, np.ndarray, Tuple[np.ndarray,np.ndarray]], 
                    h: Union[Real, np.ndarray], 
                    boundaries: Optional[Union[int, np.ndarray]],
                    t: Real = 0):
        """
            Parameters
            ----------
            n:          Union[int, np.ndarray]
                        The qubit layout of the lattice
                        If int given n -> [n, 1]

            J:          Union[Real, np.ndarray, Tuple[np.ndarray,np.ndarray]]
                        The nearest neighbour coupling
                        Tuple[np.ndarray,np.ndarray] in case of n being 2D for vertical and horizontal

            k:          Union[Real, np.ndarray, Tuple[np.ndarray,np.ndarray]]
                        The next nearest neighbour coupling
                        Tuple[np.ndarray,np.ndarray] in case of n being 2D for vertical and horizontal

            h:          Union[Real, np.ndarray]
                        The external field strength in X-direction

            boundaries: Optional[Union[int, np.ndarray]]
                        boundary conditions of the model calc if J or k are nd.arrays
                        Otherwise require to be provided

            t:          Optional[Number]
                        Final simulation time; likely not used so much when ANNNI is used with DrivenModel
        """
        assert(boundaries is not None and isinstance(J, Real) and isinstance(k, Real)),\
            "Error in ANNNI constructor: both J and k are provided as Numbers but no boundary condition was given"

        # TODO ?: Assert whether given information fits together
        # On the other hand cannot write assert for everything
        if isinstance(J, np.ndarray):
            assert(J.ndim == 1 or J.ndim == 5),\
                "Error in ANNNI constructor: J must be Number, Tuple[np.array, np.array] or np.ndarray with ndim 1 or 5, ndim of provided J is {}".format(J.ndim)

        if isinstance(k, np.ndarray):
            assert(k.ndim == 1 or k.ndim == 5),\
                "Error in ANNNI constructor: k must be Number, Tuple[np.array, np.array] or np.ndarray with ndim 2 or 5, ndim of provided k is {}".format(k.ndim)

        if isinstance(h, np.ndarray):
            assert(h.ndim == 1 or h.ndim == 2 or h.ndim == 3),\
                "Error in ANNNI constructor: h must be Number or np.ndarray with ndim 1, 2 or 3, ndim of provided h is {}".format(h.ndim)

        if isinstance(n, int):
            n = [n,1]

        self.n = np.ndarray(n)

        ## Set boundaries:
        #self.boundaries = np.array((self.n[0] - j_v.shape[1], self.n[1] - j_h.shape[2]))
        self.boundaries = self._get_boundaries(J,k, boundaries)
        _interations, _external_fields= self._get_Jh(J, k, h)

        # convert all input to np array to be sure
        #super().__init__(   "GridQubit", 
        #                    self.n,
        #                    _interations,
        #                    _external_fields,
        #                    [lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)],
        #                    [cirq.X],
        #                    t)
        self.energy_fields = ["X"]
    
    def _get_boundaries(self,J,k, boundaries):
        if boundaries is not None:
            if isinstance(boundaries, int):
                return [boundaries, 1]
            else:
                boundaries
        else:
            if isinstance(J, np.ndarray):
                # 1D case or generic J.ndim = 5 case
                if J.ndim == 1:
                    return np.array((self.n[0] - np.size(J), 1))
                else:
                    return np.array((self.n[0] - J.shape[0], self.n[1] - J.shape[1]))
            elif isinstance(J, Tuple[np.ndarray,np.ndarray]):
                # 2D case
                return np.array((self.n[0] - J[0].shape[0], self.n[1] - J[1].shape[1]))
            else:
                # J is Number and boundaries is None so that the 
                # boundaries information needs to be given in k
                if isinstance(k, np.ndarray):
                    if k.ndim == 1:
                        return np.array((self.n[0] - np.size(k), 1))
                    else:
                        return np.array((self.n[0] -k.shape[0], self.n[1] - k.shape[1]))
                else:
                    # 2D case
                    return np.array((self.n[0] - k[0].shape[0], self.n[1] - k[1].shape[1]))

    def _get_Jh(self, J, k, h):
        """
            Helper function to convert given J, k, h to 5d array as required by SpinModelFC cosntructor

            1. get all of them as arrays
            2. return _J_array = NN_array + NNN_array
        """
        if isinstance(J, Real):
            _NN_array = J*self._get_neighbour_graph(1)         
        elif isinstance(J, Real): 
            _NN_array = J 
        else:
            #J.ndim == 5 and isinstance(J, np.ndarray)
            _NN_array = J 
        
        if k.ndim == 5:
            _NNN_array = k
        else:

        if isinstance(h, Real):
            _h_array = h * np.ones((self.n[0]-self.boundaries[0],
                                    self.n[1]-self.boundaries[1],
                                    1))
        elif h.dim == 2:
            _h_array = h[:, :, np.newaxis]
        else:
            #h.ndim == 3
            _h_array = h

        return [_NN_array+_NNN_array, _h_array]
            
    def _get_neighbour_graph(   self,
                                neighbour_distance: int) -> np.ndarray:
        _neighbour_array = np.zeros((self.n[0]-self.boundaries[0],
        1))



    """
    def copy(self) -> HeisenbergFC:
        self_copy = HeisenbergFC( self.qubittype,
                self.n,
                self.j[:,:,:,:,0],
                self.j[:,:,:,:,1],
                self.j[:,:,:,:,2],
                self.h[:,:,0],
                self.h[:,:,1],
                self.h[:,:,2],
                self.t )
        
        self_copy.circuit = self.circuit.copy()
        self_copy.circuit_param = self.circuit_param.copy()
        self_copy.circuit_param_values = self.circuit_param_values.copy()
        self_copy._hamiltonian = self._hamiltonian.copy()
        
        if self.eig_val is not None: self_copy.eig_val = self.eig_val.copy()
        if self.eig_vec is not None: self_copy.eig_vec = self.eig_vec.copy()
        if self._Ut is not None: self_copy._Ut = self._Ut.copy()

        return self_copy
    """
    def energy(self) -> Tuple[np.ndarray, np.ndarray]:
        return [super().energy( self.j[:,:,:,:,0], self.h[:,:,0])]
    """
    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "qubittype": self.qubittype,
                "n": self.n,
                "j_x": self.j[:,:,:,:,0],
                "j_y": self.j[:,:,:,:,1],
                "j_z": self.j[:,:,:,:,2],
                "h_x": self.h[:,:,0],
                "h_y": self.h[:,:,1],
                "h_z": self.h[:,:,2],
                "t": self.t
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

    """