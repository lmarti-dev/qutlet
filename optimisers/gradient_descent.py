"""
This is a submodule for Optimiser()

This file is not exectuded, rather called within Ising() class when:
-set_circuit('qaoa') is called 

or functions are handed over to classical optimiser
"""
# external import
import numpy as np
import cirq
import sympy

# internal import
from .optimiser import Optimiser


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
        eps=10 ** -3,
        eta=10 ** -2,
        break_cond="iterations",
        break_param=100,
        n_print=-1,
    ):
        super().__init__()
        self._eps = eps
        self._eta = eta
        self._break_cond = break_cond
        self._break_param = break_param
        self._n_print = n_print

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
        if self._break_cond == "iterations":
            if isinstance(self._n_print, (int, np.int_)) and self._n_print > 0:
                for i in range(self._break_param):
                    # Print out
                    if not i % self._n_print:
                        # Print every n_print's step
                        wf = self._simulator.simulate(
                            self._circuit,
                            param_resolver=self._get_param_resolver(temp_cpv),
                        ).state_vector()

                        print("Steps: {}, Energy: {}".format(i, self._obj_func(wf)))
                        # print('Parameter names:  {}'.format(self.circuit_param))
                        # print('Parameter values: {}'.format(temp_cpv))
                    # End of print out

                    gradient_values = self._get_gradients(temp_cpv)

                    # Make gradient step
                    # Potentially improve this:
                    for j in range(np.size(temp_cpv)):
                        temp_cpv[j] -= self._eta * gradient_values[j]
            else:
                for i in range(self._break_param):
                    gradient_values = self._get_gradients(temp_cpv)

                    # Make gradient step
                    # Potentially improve this:
                    for j in range(np.size(temp_cpv)):
                        temp_cpv[j] -= self._eta * gradient_values[j]
        else:
            assert (
                False
            ), "Invalid break condition, received: '{}', allowed is \n \
        'iterations'".format(
                self._break_cond
            )

        # Print final result ... to be done
        wf = self._simulator.simulate(self._circuit, param_resolver=self._get_param_resolver(temp_cpv)).state_vector()
        print("Steps: {}, Energy: {}".format(i + 1, self._obj_func(wf)))
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
        joined_dict = {**{str(self.circuit_param[i]): temp_cpv[i] for i in range(n_param)}}

        # Calculate the gradients
        # Use MPi4Py/Multiporcessing HERE!!
        for j in range(n_param):
            # Simulate wavefunction at p_j + eps
            joined_dict[str(self.circuit_param[j])] = joined_dict[str(self.circuit_param[j])] + self._eps
            wf1 = self._simulator.simulate(self._circuit, param_resolver=cirq.ParamResolver(joined_dict)).state_vector()

            # Simulate wavefunction at p_j - eps
            joined_dict[str(self.circuit_param[j])] = joined_dict[str(self.circuit_param[j])] - 2 * self._eps
            wf2 = self._simulator.simulate(self._circuit, param_resolver=cirq.ParamResolver(joined_dict)).state_vector()

            # Calculate gradient
            gradient_values[j] = (self._obj_func(wf1) - self._obj_func(wf2)) / (2 * self._eps)

            # Reset dictionary
            joined_dict[str(self.circuit_param[j])] = joined_dict[str(self.circuit_param[j])] + self._eps

        # print("GradientDescent._get_gradients() does not work/is not completed")
        return gradient_values

    def _get_param_resolver(self, temp_cpv):
        joined_dict = {**{str(self.circuit_param[i]): temp_cpv[i] for i in range(np.size(self.circuit_param_values))}}
        return cirq.ParamResolver(joined_dict)


"""
#Gradient descent
  def _get_gradients(self, beta_values, gamma_values, eps = np.pi/4):
    '''
    calculate gradient for each gamma_i and beta_j from wavefunction
    #Need to write better e.g. use simulate/run sweep
    #Use MPI4py here, male a mpi4py version of this
    '''
    if self.p > 1:
      grad_betas = np.zeros(self.p)
      grad_gammas = np.zeros(self.p)
    else:
      grad_betas = 0.0
      grad_gammas = 0.0
    

    #Define dictonary once and then reuse it:
    #Bit of reduncance compared to _get_param_resolver here, 
    # maybe avoid in the future, but not straight forward as directory needs
    # to be altered
    if self.p == 1:
      #Transform beta_values, gamma_values to plain float to avoid qsim
      #IndexError
      joined_dict = {**{"b0": float(beta_values)},**{"g0": float(gamma_values)}}
    else:
      joined_dict = {**{"b" + str(i): beta_values[i] for i in range(self.p)},\
                    **{"g" + str(i): gamma_values[i] for i in range(self.p)}}
        
    #Calculate the beta gradients
    for j in range(self.p):
      #Simulate wavefunction at beta_j + eps 
      joined_dict["b" + str(j)] = joined_dict["b" + str(j)] + eps
      wf1 = self.simulator.simulate(self.circuit, \
        param_resolver = cirq.ParamResolver(joined_dict)).state_vector()
      
      #Simulate wavefunction at beta_j + eps 
      joined_dict["b" + str(j)] = joined_dict["b" + str(j)] - 2*eps
      wf2 = self.simulator.simulate(self.circuit, \
        param_resolver = cirq.ParamResolver(joined_dict)).state_vector()
      
      #Calculated gradient with simply 2eps rule and write the result into an array
      #For the gradient shift rule, this is simply the difference:
      # this is to avoid weird python IndexErrors, that occure when arrays
      # have only one element
      if abs(eps - np.pi/4) < 10**(-14):
        if self.p > 1:
          grad_betas[j] =  (self.energy(wf1)- self.energy(wf2))
        else:
          grad_betas =  (self.energy(wf1)- self.energy(wf2))
      else:
        if self.p > 1:
          grad_betas[j] =  (self.energy(wf1)- self.energy(wf2))/(2*eps)
        else:
          grad_betas  =  (self.energy(wf1)- self.energy(wf2))/(2*eps)

      #Reset dictonary
      joined_dict["b" + str(j)] = joined_dict["b" + str(j)] + eps

    #Calculate the gamma gradients
    for j in range(self.p):
      #Simulate wavefunction at gamma_j + eps 
      joined_dict["g" + str(j)] = joined_dict["g" + str(j)] + eps
      wf1 = self.simulator.simulate(self.circuit, \
        param_resolver = cirq.ParamResolver(joined_dict)).state_vector()
      
      #Simulate wavefunction at gamma_j - eps 
      joined_dict["g" + str(j)] = joined_dict["g" + str(j)] - 2*eps
      wf2 = self.simulator.simulate(self.circuit, \
        param_resolver = cirq.ParamResolver(joined_dict)).state_vector()
      
      #Calculated gradient with simply 2eps rule and write the result into an array
      #For the gradient shift rule, this is simply the difference:
      # this is to avoid weird python IndexErrors, that occur when arrays
      # have only one element
      if abs(eps - np.pi/4) < 10**(-14):
        if self.p > 1:
          grad_gammas[j] =  (self.energy(wf1)- self.energy(wf2))
        else:
          grad_gammas =  (self.energy(wf1)- self.energy(wf2))
      else:
        if self.p > 1:
          grad_gammas[j] =  (self.energy(wf1)- self.energy(wf2))/(2*eps)
        else:
          grad_gammas  =  (self.energy(wf1)- self.energy(wf2))/(2*eps)

      #Reset dictonary
      joined_dict["g" + str(j)] = joined_dict["g" + str(j)] + eps

    return grad_betas, grad_gammas

  def opt_gradient_descent(self, n_steps=1000, eps = np.pi/4, \
                          eta =10**-2, n_print = 50, dE_cut = 0): 
    # While/return
    # Include MPI4py here
    # Maybe implement other versions of gradient descent later
    # Meaning to implement different break off conditions

    #Make copy of gamma/beta_values as we don't want to update the initial arrays    
    # 2 errors to catch here:
    #     AttibuteError if gamma_values have not be set
    #     IndexError if gamma_values only contain 1 element
    try:
      temp_beta_values = self.beta_values[:]
    except AttributeError: #self.beta_values not defined
      temp_beta_values = np.random.random(self.p)
    except IndexError: #only 1 element
      temp_beta_values = self.beta_values

    try:
      temp_gamma_values = self.gamma_values[:]
    except AttributeError: #self.beta_values not defined
      temp_gamma_values = np.random.random(self.p)
    except IndexError: #only 1 element
      temp_gamma_values = self.gamma_values
    
    #print('Gammas: {}, Betas: {}'.format(temp_gamma_values, temp_beta_values))
    for i in range(n_steps):
      if not i%n_print:
        #Print every n_print's step
        wf = self.simulator.simulate(self.circuit,\
           param_resolver = self._get_param_resolver(temp_beta_values, temp_gamma_values)).state_vector()
        #print(wf)
        print('Steps: {}, Energy: {}'.format(i, self.energy(wf)))
        print('Gammas: {}, Betas: {}'.format(temp_gamma_values, temp_beta_values))

      grad_betas, grad_gammas = self._get_gradients(temp_beta_values, temp_gamma_values, eps)

      #Update gamma/beta values, watchout that eps << eta
      if self.p > 1:
        for j in range(self.p):
          temp_beta_values[j] -= eta*grad_betas[j]
          temp_gamma_values[j] -= eta*grad_gammas[j]
      else:
        #print('temp_beta_values: {}, eta: {}, grad_betas: {}'.format(temp_beta_values, eta, grad_betas))
        #print('eta*grad_betas: {}'.format(eta*grad_betas))
        temp_beta_values = temp_beta_values - eta*grad_betas
        temp_gamma_values = temp_gamma_values - eta*grad_gammas
    #Print final result
    wf = self.simulator.simulate(self.circuit,\
           param_resolver = self._get_param_resolver(temp_beta_values, temp_gamma_values)).state_vector()
    print('Minimised energy: {}, Betas: {}, Gammas {}'.format(self.energy(wf), temp_beta_values, temp_gamma_values ))

    #yield? or overwrite self.gamma/beta_values?
    #return temp_gamma_values, temp_beta_values
    self.beta_values = temp_beta_values
    self.gamma_values = temp_gamma_values
"""
