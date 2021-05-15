"""
    This should become the future abstract optimiser parentclass

What does a optimiser do?:
    Abstractly (in general):
        - What does a optimiser do abstractly?
            -> Takes high-dim/non trivial function f(x) at x_i and gives a new x_i+1 to max/min f(x_i+1)
            -> Iterate ; terminate/finish with some break condition

    Abstractly (for QC):
        -Mostly the objectiv function f(x) is the energy
        -Return/Update circuit parameters 

What does a optimiser need for that?:
    -Objectiv function f(x)/energy -> referenc to external energy function, energy gets wavefunction
    -parametrised circuit to optimise + quibits array
    -circuit parameters based on which the optimisation is done
    -simulator to simulate wavefunction

"""
# external import
import abc

from fauvqe.objectives.objective import Objective


class Optimiser(abc.ABC):
    """
    Args:
        obj_func()          :   objectiv function f(x)/energy
        qubits              :   qubit array/ordering for parametrised circuit
        simulator           :   Classical quantum simulator to simulate circuit
        circuit             :   parametrised circuit
        circuit_param       :   sympy.Symbols for parametrised circuit

        circuit_param_values:   current/initial values of circuit parameters
                                ->To be updates

    These e.g. exist in Ising() as ising_obj.qubits etc..., but copy/view seems most reasonable
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def optimise(self, objective: Objective):
        """
        Idea:
            -Each optimiser should have function "optimise"
            -Definition here just to ensure existence/uniform naming
            -one can simply call the specific optimsier class e.g. optimser_obj = GradientDescent(obj_func, qubits,simulator, circuit, circuit_param, circuit_param_values)
            -And then optimser_obj.optimise() should run/work via default optimiser parameters if necessary

        Run optimiser until break condition is fullfilled
        """
        raise NotImplementedError()  # pragma: no cover
