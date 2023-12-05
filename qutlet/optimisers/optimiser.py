"""This should become the future abstract optimiser parentclass

What does a optimiser do?:

Abstractly (in general):
    - What does a optimiser do abstractly?
        -> Takes high-dim/non trivial function f(x) at x_i and gives a new x_i+1 to max/min f(x_i+1)
        -> Iterate ; terminate/finish with some break condition

Abstractly (for QC):
    -Mostly the objectiv function f(x) is the energy
    -Return/Update circuit parameters

"""
import abc

from fauvqe.objectives.objective import Objective
from fauvqe.restorable import Restorable


class Optimiser(Restorable):
    """Optimiser"""

    def __init__(self):
        pass

    @abc.abstractmethod
    def optimise(self, objective: Objective):
        """Run optimiser until break condition is fulfilled

        Parameters
        ----------
        objective
        """
        raise NotImplementedError()  # pragma: no cover
