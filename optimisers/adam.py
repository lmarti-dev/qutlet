"""
This is the ADAM-submodule for Optimiser()

This file is not exectuded, rather called within Ising() class when:
-set_circuit('qaoa') is called 

or functions are handed over to classical optimiser

Implementation following arXiv 1412.6980

Algorithm 1: Adam, our proposed algorithm for stochastic optimization. See section 2 for details,
and for a slightly more efﬁcient (but less clear) order of computation. g 2
t indicates the elementwise square g_t  g_t . Good default settings for the tested machine learning problems are α = 0.001,
β_1 = 0.9, β_2 = 0.999 and eps = 10− 8 . All operations on vectors are element-wise. With β1 and β2 we denote β_1 and β_2 to the power t.

Require: α: Stepsize
Require: β1 , β2 ∈ [0, 1): Exponential decay rates for the moment estimates
Require: f(θ): Stochastic objective function with parameters θ
Require: θ 0 : Initial parameter vector
    m_0 ← 0 (Initialize 1 st moment vector)
    v_0 ← 0 (Initialize 2 nd moment vector)
    t ← 0   (Initialize timestep)
    while θ t not converged do
        t ← t + 1
        g_t ← ∇_θ f_t (θ_{t−1} )                    (Get gradients w.r.t. stochastic objective at timestep t)
        m_t ← β_1 · m_t + (1 − β_1 ) · g_t      (Update biased ﬁrst moment estimate)
        v_t ← β_2 · v_t + (1 − β_2 ) · g^2_t    (Update biased second raw moment estimate)
        \hat m_t ← m_t /(1 − β^t_1 )                (Compute bias-corrected ﬁrst moment estimate)
        \hat v_t ← v_t /(1 − β^t_2 )                (Compute bias-corrected second raw moment estimate)
        θ_t ← θ_{t−1} − α · \hat m_t /( (\hat v_t)^0.5 + eps) (Update parameters)
    end while
return θ t (Resulting parameters)

Potential MUCH BETTER alternative option: load from scipy??

"""
# external import
import numpy as np
import cirq
import sympy
import joblib
import multiprocessing

# internal import
from fauvqe.optimisers import Optimiser
class ADAM(Optimiser):
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
      eps_2       : epsilon for adam
      b_1, b_2    : beta_1, beta_2 for adam

      break_cond  :default 'iterations', but also e.g. change in obj_func etc.
      break_param : e.g amount of steps of iteration

      n_print : print out after n steps, default -1

    Try to include/implement MPI4py or multiprocessing here!
  """
  def __init__(self, obj_func, qubits,simulator, circuit, circuit_param, circuit_param_values,
                eps=10**-3,eps_2=10**-8, a = 10**-2,b_1 = 0.9, b_2 = 0.999, break_cond='iterations', break_param=100, n_print=-1):
    super().__init__(obj_func, qubits,simulator, circuit, circuit_param, circuit_param_values)
    self.eps=eps;
    self.eps_2=eps_2;
    self.a = a;
    self.b_1=b_1;
    self.b_2=b_2;
    self.break_cond=break_cond;
    self.break_param=break_param;
    self.n_print=n_print;
    self.step = 0;
    self._v_t = np.zeros(np.shape(circuit_param_values));
    self._m_t = np.zeros(np.shape(circuit_param_values));

  def optimise(self):
    """
      Run optimiser until break condition is fullfilled

      1.make copies of param_values (to not accidentially overwrite)
      2.Do steps until break condition
        Tricky need to generalise _get_param_resolver method which also affects _get_gradients
      3.Update self.circuit_param_values = temp_cpv

      NEED FOR IMPROVEMENT:
        - allow to update hyperparameters via attribute pass to optimise function
        - Possibly by ** args or python dictionary???
    """
    #1.make copies of param_values (to not accidentially overwrite)
    temp_cpv = self.circuit_param_values;

    #Do step until break condition
    if self.break_cond == 'iterations':
      if isinstance(self.n_print, (int, np.int_)) and self.n_print > 0:
        for i in range(self.break_param):
          #Print out
          if not i%self.n_print:
            #Print every n_print's step
            wf = self.simulator.simulate(self.circuit,\
            param_resolver = self._get_param_resolver(temp_cpv)).state_vector()
        
            print('Steps: {}, Energy: {}'.format(i, self.obj_func(wf)))
            #print('Parameter names:  {}'.format(self.circuit_param))
            #print('Parameter values: {}'.format(temp_cpv))
          #End of print out
          
          #Make ADAM step
          temp_cpv = self._ADAM_step(temp_cpv)
      else:
        for i in range(self.break_param):
            #Make ADAM step
            temp_cpv = self._ADAM_step(temp_cpv)
    else:
      assert False, "Invalid break condition, received: '{}', allowed is \n \
        'iterations'".format(self.break_cond )

    #Print final result ... to be done
    wf = self.simulator.simulate(self.circuit,\
           param_resolver = self._get_param_resolver(temp_cpv)).state_vector()
    print('Steps: {}, Energy: {}'.format(i+1, self.obj_func(wf)))
    print('Parameter names:  {}'.format(self.circuit_param))
    print('Parameter values: {}'.format(temp_cpv))

    #Return/update circuit_param_values
    self.circuit_param_values = temp_cpv;

  def _ADAM_step(self, temp_cpv):
        '''
            t ← t + 1
            g_t ← ∇_θ f_t (θ_{t−1} )                    (Get gradients w.r.t. stochastic objective at timestep t)
            m_t ← β_1 · m_t + (1 − β_1 ) · g_t      (Update biased ﬁrst moment estimate)
            v_t ← β_2 · v_t + (1 − β_2 ) · g^2_t    (Update biased second raw moment estimate)
            \hat m_t ← m_t /(1 − β^t_1 )                (Compute bias-corrected ﬁrst moment estimate)
            \hat v_t ← v_t /(1 − β^t_2 )                (Compute bias-corrected second raw moment estimate)
            θ_t ← θ_{t−1} − α · \hat m_t /( (\hat v_t)^0.5 + eps) (Update parameters)

            Alternative for last 3 lines:
            α_t = α · (1 − β^t_2)^0.5/(1 − β^t_1) 
            θ_t ← θ_{t−1} − α_t · m_t  /( (v_t)^0.5 + eps)

        '''
        self.step += 1
        gradient_values = self._get_gradients(temp_cpv)
        print(gradient_values)
        #print(np.shape(gradient_values))
        self._m_t = self.b_1 * self._m_t + (1-self.b_1)*gradient_values
        self._v_t = self.b_2 * self._v_t + (1-self.b_2)*gradient_values**2
        temp_cpv -= self.a * (1 - self.b_2**self.step)**0.5/(1 - self.b_1**self.step) \
                    *self._m_t/( self._v_t**0.5 + self.eps_2)
        return temp_cpv


  def _get_gradients(self, temp_cpv):
    n_param = np.size(self.circuit_param_values);
    gradient_values = np.zeros(n_param);

    #Create joined dictionary from 
    #self.circuit_param <- need their name list
    #     create name array?
    #Use temp_cpv not self.circuit_param_values here
    joined_dict = {**{str(self.circuit_param[i]): temp_cpv[i] for i in range(n_param)}};
    
    #Calculate the gradients
    #Use MPi4Py/Multiporcessing HERE!!
    for j in range(n_param):
      #Simulate wavefunction at p_j + eps 
      joined_dict[str(self.circuit_param[j])] = joined_dict[str(self.circuit_param[j])] + self.eps;
      wf1 = self.simulator.simulate(self.circuit, \
        param_resolver = cirq.ParamResolver(joined_dict)).state_vector()

      #Simulate wavefunction at p_j - eps 
      joined_dict[str(self.circuit_param[j])] = joined_dict[str(self.circuit_param[j])] - 2*self.eps;
      wf2 = self.simulator.simulate(self.circuit, \
        param_resolver = cirq.ParamResolver(joined_dict)).state_vector()

      #Calculate gradient
      gradient_values[j] =  (self.obj_func(wf1)- self.obj_func(wf2))/(2*self.eps);

      #Reset dictionary
      joined_dict[str(self.circuit_param[j])] = joined_dict[str(self.circuit_param[j])] + self.eps;

    #print("GradientDescent._get_gradients() does not work/is not completed")
    return gradient_values
  
  def _get_param_resolver(self, temp_cpv):
    joined_dict = {**{str(self.circuit_param[i]): temp_cpv[i] for i in range(np.size(self.circuit_param_values))}};
    return cirq.ParamResolver(joined_dict)

#########################################################################
#                                                                       #
#           Add experimental different paralelisation options           #
#                                                                       #
#########################################################################

  def optimise_joblib(self, n_jobs = 'default'):
    if n_jobs == 'default':
        try:
            _n_jobs = np.divmod(multiprocessing.cpu_count(), ising_obj.simulator_options['t'])[0]
        except:
            _n_jobs = multiprocessing.cpu_count()

    #1.make copies of param_values (to not accidentially overwrite)
    temp_cpv = self.circuit_param_values;

    #Do step until break condition
    if self.break_cond == 'iterations':
      if isinstance(self.n_print, (int, np.int_)) and self.n_print > 0:
        for i in range(self.break_param):
          #Print out
          if not i%self.n_print:
            #Print every n_print's step
            wf = self.simulator.simulate(self.circuit,\
            param_resolver = self._get_param_resolver(temp_cpv)).state_vector()
            print('Steps: {}, Energy: {}'.format(i, self.obj_func(wf)))
          #End of print out
          
          #Make ADAM step
          temp_cpv = self._ADAM_step_joblib(temp_cpv, _n_jobs)
      else:
        for i in range(self.break_param):
            #Make ADAM step
            temp_cpv = self._ADAM_step_joblib(temp_cpv, _n_jobs)
    else:
      assert False, "Invalid break condition, received: '{}', allowed is \n \
        'iterations'".format(self.break_cond )

    #Print final result ... to be done
    wf = self.simulator.simulate(self.circuit,\
           param_resolver = self._get_param_resolver(temp_cpv)).state_vector()
    print('Steps: {}, Energy: {}'.format(i+1, self.obj_func(wf)))
    print('Parameter names:  {}'.format(self.circuit_param))
    print('Parameter values: {}'.format(temp_cpv))

    #Return/update circuit_param_values
    self.circuit_param_values = temp_cpv;

  def _ADAM_step_joblib(self, temp_cpv, _n_jobs):
        '''
            t ← t + 1
            g_t ← ∇_θ f_t (θ_{t−1} )                    (Get gradients w.r.t. stochastic objective at timestep t)
            m_t ← β_1 · m_t + (1 − β_1 ) · g_t      (Update biased ﬁrst moment estimate)
            v_t ← β_2 · v_t + (1 − β_2 ) · g^2_t    (Update biased second raw moment estimate)
            \hat m_t ← m_t /(1 − β^t_1 )                (Compute bias-corrected ﬁrst moment estimate)
            \hat v_t ← v_t /(1 − β^t_2 )                (Compute bias-corrected second raw moment estimate)
            θ_t ← θ_{t−1} − α · \hat m_t /( (\hat v_t)^0.5 + eps) (Update parameters)

            Alternative for last 3 lines:
            α_t = α · (1 − β^t_2)^0.5/(1 − β^t_1) 
            θ_t ← θ_{t−1} − α_t · m_t  /( (v_t)^0.5 + eps)

        '''
        self.step += 1
        gradient_values = np.array(self._get_gradients_joblib(temp_cpv, _n_jobs))
        #gradient_values = np.array(gradient_values)
        print(gradient_values)
        #print(np.shape(gradient_values))
        self._m_t = self.b_1 * self._m_t + (1-self.b_1)*gradient_values
        self._v_t = self.b_2 * self._v_t + (1-self.b_2)*gradient_values**2
        temp_cpv -= self.a * (1 - self.b_2**self.step)**0.5/(1 - self.b_1**self.step) \
                    *self._m_t/( self._v_t**0.5 + self.eps_2)
        return temp_cpv

  def _get_gradients_joblib(self, temp_cpv, _n_jobs):
    n_param = np.size(self.circuit_param_values);
    joined_dict = {**{str(self.circuit_param[i]): temp_cpv[i] for i in range(n_param)}};
    return joblib.Parallel(n_jobs = _n_jobs, backend='threading')\
            (joblib.delayed(self._get_single_gradient_joblib)(joined_dict.copy(), j) for j in range(n_param))

  def _get_single_gradient_joblib(self, joined_dict, j):
    #Simulate wavefunction at p_j + eps 
      joined_dict[str(self.circuit_param[j])] = joined_dict[str(self.circuit_param[j])] + self.eps;
      wf1 = self.simulator.simulate(self.circuit, \
        param_resolver = cirq.ParamResolver(joined_dict)).state_vector()

      #Simulate wavefunction at p_j - eps 
      joined_dict[str(self.circuit_param[j])] = joined_dict[str(self.circuit_param[j])] - 2*self.eps;
      wf2 = self.simulator.simulate(self.circuit, \
        param_resolver = cirq.ParamResolver(joined_dict)).state_vector()

      #Calculate gradient + return
      return (self.obj_func(wf1)- self.obj_func(wf2))/(2*self.eps);
      #No need to reset Reset dictionary
