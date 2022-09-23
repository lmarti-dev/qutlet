import scipy
from fauvqe.optimisers.optimiser import Optimiser
from fauvqe.objectives.objective import Objective
from fauvqe.optimisers.optimisation_result import OptimisationResult

from scipy.optimize import minimize,OptimizeResult
import numpy as np
from typing import Dict
import cirq

# available optimizers:
# ‘Nelder-Mead’ 
# ‘Powell’ 
# ‘CG’ 
# ‘BFGS’ 
# ‘Newton-CG’ 
# ‘L-BFGS-B’ 
# ‘TNC’ 
# ‘COBYLA’ 
# ‘SLSQP’ 
# ‘trust-constr’
# ‘dogleg’ 
# ‘trust-ncg’ 
# ‘trust-exact’ 
# ‘trust-krylov’

class ScipyOptimisers(Optimiser):
    def __init__(self,minimize_options={"method": "L-BFGS-B"},initial_state=None,initial_params="random"):
        self.minimize_options = minimize_options
        self.initial_state=initial_state
        self.initial_params=initial_params
        super().__init__()
    def simulate(self,x):
        wf = self._objective.simulate(param_resolver=self.get_param_resolver(x),initial_state=self.initial_state)
        return wf
    def fun(self,x):
        wf = self.simulate(x)
        return self._objective.evaluate(wavefunction=wf)
    def optimise(self, objective: Objective):
        self._objective = objective
        self._fauvqe_res= OptimisationResult(self._objective)
        if self.initial_params == "random":
            x0=np.random.rand(*np.shape(self._objective.model.circuit_param_values.view()))
        elif self.initial_params == "zeros":
            x0 = np.zeros(np.shape(self._objective.model.circuit_param_values.view()))
        def process_step(xk):
            wf = self.simulate(xk)
            self._fauvqe_res.add_step(params=xk,
                                    wavefunction=wf,
                                    objective=self.fun(xk))
            # print(self._fauvqe_res)

        scipy_res = minimize(self.fun,x0,**self.minimize_options,callback=process_step)
        # add the last step when the simulation is done
        x_final=scipy_res.x
        process_step(x_final)
        return self._fauvqe_res

    def get_param_resolver(self,x):
        joined_dict = {**{str(self._objective.model.circuit_param[i]): x[i] for i in range(len(self._objective.model.circuit_param))}}
        return cirq.ParamResolver(joined_dict)
    def from_json_dict(self) -> Dict:
        pass
    
    def to_json_dict(self) -> Dict:
        pass
            
