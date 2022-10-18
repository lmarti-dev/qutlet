import cirq
import numpy as np

from fauvqe.models.abstractmodel import AbstractModel
import fauvqe.utils_cirq as cqutils
import fauvqe.utils as utils
import sympy
import openfermion as of
import itertools
import scipy
from math import prod


def set_from_generators(generators: "list[cirq.PauliSum]",order=3,make_anti_hermitian:bool=True):
    paulisum_set=[]
    for o in range(order):
        combinations=itertools.combinations(range(len(generators)),o+1)
        for indices in combinations:
            if make_anti_hermitian:
                paulisum_set.append(prod([cqutils.make_pauli_sum_hermitian(generators[iii],anti=True) for iii in indices]))
            else:
                paulisum_set.append(prod([generators[iii] for iii in indices]))
    return paulisum_set




def pauli_grad(model:AbstractModel,operator: cirq.PauliSum,wf:np.ndarray,eps:float=1e-6,anti:bool=True) -> float:
    # <p|[H,A(k)]|p> = <p|(HA(k) - A(k)H)|p> = dE/dk
    # finite diff (f(theta + eps) - f(theta - eps))/ 2eps but theta = 0
    # if A is anti hermitian
    # (<p|exp(-eps*operator) H exp(eps*operator)|p> - <p|exp(eps*operator) H exp(-eps*operator)|p>)/2eps
    # if A is hermitian the theta needs to be multiplied by 1j
    ham = model.hamiltonian

    qmap=cqutils.qmap(model)
    wfexp = scipy.sparse.linalg.expm_multiply(A=operator.toarray(),B=wf,start=-eps,stop=eps,num=2,endpoint=True)
    grad_left = ham.expectation_from_state_vector(wfexp[0,:],qubit_map=qmap)
    grad_right = ham.expectation_from_state_vector(wfexp[1,:],qubit_map=qmap)
    
    return np.abs((grad_left-grad_right)/(2*eps))

def exp_from_pauli_sum(pauli_sum: cirq.PauliSum, theta=0):
    return cirq.PauliSumExponential(pauli_sum_like=pauli_sum,exponent=theta)

def get_best_gate(model:AbstractModel,paulisum_set:list,param_name:str,tol:float,initial_wf=None):
    if initial_wf is not None:
        wf = initial_wf
    else:
        wf = model.simulator.simulate(model.circuit,param_resolver=cqutils.get_param_resolver(model=model,param_values=model.circuit_param_values)).final_state_vector
    grad_values = [pauli_grad(model=model,operator=ps,wf=wf) for ps in paulisum_set]
    print(grad_values)
    max_index = np.argmax(grad_values)
    best_ps = paulisum_set[max_index]
    if grad_values[max_index] < tol and tol is not None:
        # iteration process is done, gradient < tol, or if stopping is set with max depth, continue
        return None
    else:
        theta = sympy.Symbol("theta_{param_name}".format(param_name=param_name))
        return exp_from_pauli_sum(pauli_sum=best_ps,theta=theta),theta


def circuit_iterating_step(model:AbstractModel,paulisum_set:list,tol:float=1e-15,default_param_val:float=0,initial_wf=None):
    res = get_best_gate(model=model,paulisum_set=paulisum_set,param_name=utils.random_name(lenw=3,Nwords=1),tol=tol,initial_wf=initial_wf)
    if res is None:
        return True
    else:
        best_ps_exp,theta = res
        model.circuit.append(best_ps_exp)
        model.circuit_param.append(theta)
        model.circuit_param_values.append(default_param_val)
        return False
        

