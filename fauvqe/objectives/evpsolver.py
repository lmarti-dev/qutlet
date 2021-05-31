"""
    Implementation of an exact solver for an AbstractModel object.

    This works locally for up to 14 qubits and 
    on the AMD 7502 EPYC nodes for up to 16 qubits (approx 45 min.)
"""
import scipy
import numpy as np

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel


class EVPSolver(Objective):
    """
    Eigen-Value-Problem (EVP) solver

    This class implements as objective the expectation value of the energies
    of the linked model.

    Parameters
    ----------
    model: AbstractModel
        The linked model
    val:    eigen values, normalised by qubit number to keep compareability
    vec:    eigen vector

    different solvers, possibly use import lib
        "numpy"         np.linalg.eigh
        "scipy"         scipy.linalg.eigh
        "scipy.sparse"  scipy.sparse.linalg.eigsh
        + Add more hardwareefficent solver
        + Set usful defaults. 
    parameters to pass on to the solver
        implement as dict
        e.g. k = 2

    Methods
    ----------
    evaluate() : 
        Calculates
        ---------
        self.vec and self.val
    """

    def __init__(self, model: AbstractModel):
        super().__init__(model)
        self.hamiltonian = model.hamiltonian
        self.__n = np.size(model.qubits)

    def evaluate(self, solver = "scipy.sparse", solver_options: dict = {}):
        if solver == "numpy":
            self.val, self.vec =  np.linalg.eigh(self.hamiltonian.matrix())
            # Normalise eigenvalues
            self.val /= self.__n           
        elif solver == "scipy":
            self.solver_options = { "check_finite": False, 
                                    "subset_by_index": [0, 1]}
            self.solver_options.update(solver_options)           
            self.val, self.vec = scipy.linalg.eigh(
                self.hamiltonian.matrix(), 
                **self.solver_options,
                )
            # Normalise eigenvalues
            self.val /= self.__n
        elif solver == "scipy.sparse":
            self.solver_options = { "k": 2,
                                    "which": 'SA'}
            self.solver_options.update(solver_options)
            self.val, self.vec = scipy.sparse.linalg.eigsh(
                self.hamiltonian.matrix(), 
                **self.solver_options,
                )
            # Normalise eigenvalues
            self.val /= self.__n
        else:
            assert False, "Invalid simulator option, received {}, allowed is 'numpy', 'scipy', 'scipy.sparse'".format(
                solver
            )

    def __repr__(self) -> str:
        return "<EVPSolver>"
