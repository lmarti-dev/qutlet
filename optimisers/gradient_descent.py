"""
This is a submodule for Optimiser()

This file is not exectuded, rather called within Ising() class when:
-set_circuit('qaoa') is called 

or functions are handed over to classical optimiser
"""
import numpy as np
import cirq
import sympy

from fauvqe.optimisers import Optimiser
from fauvqe.objectives import Objective


class GradientDescent(Optimiser):
    """
    Args from parrent class:
        obj_func()          :   objectiv function f(x)/energy
        qubits              :   qubit array/ordering for parametrised circuit
        simulator           :   Classical quantum simulator to simulate circuit
        circuit             :   parametrised circuit
        circuit_param       :   sympy.Symbols for parametrised circuit
        circuit_param_values:   current/initial values of circuit parameters
                                    ->To be updates

    Additional Args (give default values):
      eps         :eps for gradient
      eta         :velocity of gd method

      break_cond  :default 'iterations', but also e.g. change in obj_func etc.
      break_param : e.g amount of steps of iteration

      n_print : print out after n steps, default -1

    Try to include/implement MPI4py or multiprocessing here!
    """

    def __init__(
        self,
        objective: Objective,
        qubits,
        simulator,
        circuit,
        circuit_param,
        circuit_param_values,
        eps=10 ** -3,
        eta=10 ** -2,
        break_cond="iterations",
        break_param=100,
        n_print=-1,
    ):
        super().__init__(
            objective, qubits, simulator, circuit, circuit_param, circuit_param_values
        )
        self.eps = eps
        self.eta = eta
        self.break_cond = break_cond
        self.break_param = break_param
        self.n_print = n_print

    def optimise(self):
        """
        Run optimiser until break condition is fullfilled

        1.make copies of param_values (to not accidentially overwrite)
        2.Do steps until break condition
          Tricky need to generalise _get_param_resolver method which also affects _get_gradients
        3.Update self.circuit_param_values = temp_cpv

        NEED FOR IMPROVEMENT:
          - allow to update hyperparameters via attribute pass to optimise function
          - Possibly by ** args or python dictionary
        """
        # 1.make copies of param_values (to not accidentially overwrite)
        temp_cpv = self.circuit_param_values

        # Do step until break condition
        if self.break_cond == "iterations":
            if isinstance(self.n_print, (int, np.int_)) and self.n_print > 0:
                for i in range(self.break_param):
                    # Print out
                    if not i % self.n_print:
                        # Print every n_print's step
                        wf = self.simulator.simulate(
                            self.circuit,
                            param_resolver=self._get_param_resolver(temp_cpv),
                        ).state_vector()

                        print(
                            "Steps: {}, Energy: {}".format(
                                i, self.objective.evaluate(wf)
                            )
                        )
                        # print('Parameter names:  {}'.format(self.circuit_param))
                        # print('Parameter values: {}'.format(temp_cpv))
                    # End of print out

                    gradient_values = self._get_gradients(temp_cpv)

                    # Make gradient step
                    # Potentially improve this:
                    for j in range(np.size(temp_cpv)):
                        temp_cpv[j] -= self.eta * gradient_values[j]
            else:
                for i in range(self.break_param):
                    gradient_values = self._get_gradients(temp_cpv)

                    # Make gradient step
                    # Potentially improve this:
                    for j in range(np.size(temp_cpv)):
                        temp_cpv[j] -= self.eta * gradient_values[j]
        else:
            assert (
                False
            ), "Invalid break condition, received: '{}', allowed is \n \
        'iterations'".format(
                self.break_cond
            )

        # Print final result ... to be done
        wf = self.simulator.simulate(
            self.circuit, param_resolver=self._get_param_resolver(temp_cpv)
        ).state_vector()
        print("Steps: {}, Energy: {}".format(i + 1, self.objective.evaluate(wf)))
        print("Parameter names:  {}".format(self.circuit_param))
        print("Parameter values: {}".format(temp_cpv))

        # Return/update circuit_param_values
        self.circuit_param_values = temp_cpv

    def _get_gradients(self, temp_cpv):
        n_param = np.size(self.circuit_param_values)
        gradient_values = np.zeros(n_param)

        # Create joined dictionary from
        # self.circuit_param <- need their name list
        #     create name array?
        # Use temp_cpv not self.circuit_param_values here
        joined_dict = {
            **{str(self.circuit_param[i]): temp_cpv[i] for i in range(n_param)}
        }

        # Calculate the gradients
        # Use MPi4Py/Multiporcessing HERE!!
        for j in range(n_param):
            # Simulate wavefunction at p_j + eps
            joined_dict[str(self.circuit_param[j])] = (
                joined_dict[str(self.circuit_param[j])] + self.eps
            )
            wf1 = self.simulator.simulate(
                self.circuit, param_resolver=cirq.ParamResolver(joined_dict)
            ).state_vector()

            # Simulate wavefunction at p_j - eps
            joined_dict[str(self.circuit_param[j])] = (
                joined_dict[str(self.circuit_param[j])] - 2 * self.eps
            )
            wf2 = self.simulator.simulate(
                self.circuit, param_resolver=cirq.ParamResolver(joined_dict)
            ).state_vector()

            # Calculate gradient
            gradient_values[j] = (
                self.objective.evaluate(wf1) - self.objective.evaluate(wf2)
            ) / (2 * self.eps)

            # Reset dictionary
            joined_dict[str(self.circuit_param[j])] = (
                joined_dict[str(self.circuit_param[j])] + self.eps
            )

        # print("GradientDescent._get_gradients() does not work/is not completed")
        return gradient_values

    def _get_param_resolver(self, temp_cpv):
        joined_dict = {
            **{
                str(self.circuit_param[i]): temp_cpv[i]
                for i in range(np.size(self.circuit_param_values))
            }
        }
        return cirq.ParamResolver(joined_dict)
