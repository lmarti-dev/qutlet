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
    val:    eigen values
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

    def evaluate(self, solver = "scipy.sparse", solver_options: dict = {}):
        if solver == "numpy":
            raise NotImplementedError() 
        elif solver == "scipy":
            raise NotImplementedError() 
        elif solver == "scipy.sparse":
            self.solver_options = {"k": 2, "which": 'SA'}
            self.solver_options.update(solver_options)
            self.val, self.vec = scipy.sparse.linalg.eigsh(
                self.hamiltonian.matrix(), 
                k = 2 , which = 'SA',
                )
        else:
            assert False, "Invalid simulator option, received {}, allowed is 'qsim', 'cirq'".format(
                simulator_name
            )

    def __repr__(self) -> str:
        return "<EVPSolver>"
