"""
    Use QAOA Ising to test if gradient descent optimiser works
    1.create Ising object + simple 4 qubit QAOA?
    2. set_optimiser
    3.ising_obj.optimise()

    Later: possibly put into test class 

    26.03.2020:
        test_optimse currently fails as energy() is jZZ-hX energy in optimiser
        but one needs to give in this case the optimiser energy(.., field='Z')
        Needs to be fixed by generalisation of Ising.set_simulator()

"""
# external imports
import pytest
import numpy as np
from typing import Optional

import cirq

# internal imports
from fauvqe import Ising, GradientDescent, ExpectationValue, UtCost
from fauvqe import IsingXY, AbstractExpectationValue, SpinModel

class Mock_UtCost(UtCost):
    def __init__(self, model, t, initial_wavefunctions ):
        super().__init__(model, t, initial_wavefunctions=initial_wavefunctions)
        delattr(self, '_time_steps')
        self.evaluate = self._evaluate

    def simulate( self, 
                param_resolver = cirq.ParamResolver(), 
                initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        if initial_state is None:
            initial_state = self._initial_wavefunctions
        
        output_state = np.empty(shape=(1, *initial_state.shape), dtype = self._dtype)
        
        if len(initial_state) == self._N:
            ini = initial_state
        else:
            ini = initial_state[0]

        wf = self._model.simulator.simulate(
                self._model.circuit,
                param_resolver=param_resolver,
                initial_state=ini,
                ).state_vector()
        output_state[0] = wf/np.linalg.norm(wf)

        #Run through all states given in initial state
        for i_state in range(1,round(np.size(initial_state)/self._N)):
            wf = self._model.simulator.simulate(
                self._model.circuit,
                param_resolver=param_resolver,
                initial_state=initial_state[i_state],
                ).state_vector()
            output_state[0, i_state] = wf/np.linalg.norm(wf)

        return np.squeeze(output_state)
    
    def _evaluate( self,
                       wavefunction: np.ndarray, 
                       options: dict = {}) -> np.float64:
        cost = 0
        if options.get('state_indices') is None:
            options['state_indices'] = range(np.size(self._output_wavefunctions[0,:,0]))

        options['time_indices'] = range(1)
        
        if wavefunction.ndim == 1:
            # This assumes a single state vector is given
            return 1- abs(np.vdot(wavefunction, self._output_wavefunctions[0,0,:]) )**self._exponent
        elif len(options.get('time_indices')) == 1 and wavefunction.ndim == 2:
            #This covers the expected behaviour for using UTCost together with a batch of random vectors and 1 final simulation time
            _tmp = [np.vdot(wavefunction[i], self._output_wavefunctions[0,i,:]) for i in range(np.size(wavefunction[:,0]))]
            return 1- (abs(np.sum(_tmp))/len(wavefunction[:,0]))**self._exponent
        else:
        # Past comment: seems wrong? -> No test to that shows that?
        # This fails UtCost self-consistency tests:
         #This covers using UTCost together with a batch of more than one final simulation time
         #Note here the inconsistent use of wavefunction compared to all other objects as
         # dim 1: time steps dim 2: batc vector dim3: vector entries
            for step in options.get('time_indices'):
                cost += (1/len(options.get('state_indices')))*np.sum(1 - (abs(np.sum(np.conjugate(wavefunction[step])*
                                            self._output_wavefunctions[step][options.get('state_indices')], 
                                            axis=1)))**self._exponent )
        return (1 /len(options.get('time_indices'))) * cost

#This test misses a real assert
@pytest.mark.higheffort
def test_set_optimiser():
    ising_obj = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    #ising_obj = SpinModel(
    #    "GridQubit", 
    #    [1, 2], 
    #    [np.ones((0, 2))], 
    #    [np.ones((1, 1))], 
    #    [np.ones((1, 2))], 
    #    [lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)], 
    #    [cirq.X]
    #)
    ising_obj.set_circuit("qaoa", {"p": 1})
    gd = GradientDescent()
    obj = ExpectationValue(ising_obj)
    #obj = AbstractExpectationValue(ising_obj)
    gd.optimise(obj)

    #Add pro forma assert:
    assert True


# This is potentially a higher effort test:
# This covers line 223 in gradient descent but pytest does not show it...
@pytest.mark.higheffort
def test_optimise():
    ising_obj = Ising(
        "GridQubit",
        [2, 2],
        0.1 * np.ones((1, 2)),
        0.5 * np.ones((2, 1)),
        0.2 * np.ones((2, 2)),
        "Z"
    )
    ising_obj.set_circuit("qaoa", {"p": 2, "H_layer": False})
    ising_obj.set_circuit_param_values(0.3 * np.ones(np.size(ising_obj.circuit_param)))
    eta = 2e-2
    gd = GradientDescent({
        'break_param':25,
        'eta':eta,
        'use_progress_bar': True, 
    })
    obj = ExpectationValue(ising_obj)
    res = gd.optimise(obj)

    final_step = res.get_latest_step()

    assert -0.5 > final_step.objective - eta
    # Result smaller than -0.5 up to eta

@pytest.mark.higheffort
@pytest.mark.parametrize(
    "cost_class, symmetric_gradient",
    [
        (
            UtCost,
            False,
        ),
        (
            Mock_UtCost,
            True,
        ),
        (
            Mock_UtCost,
            False,
        ),
    ]
)
def test_optimise_batch(cost_class, symmetric_gradient):
    t=0.5
    ising = Ising("GridQubit", [1, 4], np.ones((0, 4)), np.ones((1, 4)), np.ones((1,4)), "X", t)
    ising.set_Ut()
    ising.set_circuit("hea", {
        "parametrisation": "joint", #"layerwise",
        "p": 3,
    })
    ising.set_circuit_param_values(-(2/np.pi)*t/3 *np.ones(np.size(ising.circuit_param)))
    
    bsize = 1
    initial_rands= (np.random.rand(bsize, 16)).astype(np.complex128)
    initials = np.zeros(initial_rands.shape, dtype=np.complex64)
    for k in range(bsize):
        initials[k, :] = initial_rands[k, :] / np.linalg.norm(initial_rands[k, :])
    
    objective = cost_class(ising, t, initial_wavefunctions = initials)
    
    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values), initial_state=initials[0]
    )
    trotter_cost = ( objective.evaluate(np.array([wavefunction]), options={'indices': [0]}) )
    print(trotter_cost)
    gd = GradientDescent({
        'break_param': 10,
        'batch_size': 1,
        'eps': 1e-5,
        'eta': 1e-2,
        'symmetric_gradient': symmetric_gradient
    })
    print(objective.model.circuit_param_values.view())

    res = gd.optimise(objective)
    print(res.get_latest_step().params)
    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(res.get_latest_step().params), initial_state=initials[0]
    )
    var_cost = (objective.evaluate(np.array([wavefunction]), options={'indices': [0]}))
    print(var_cost)
    assert var_cost/10 < trotter_cost

def test_json():
    t=0.1
    j_v = np.ones((0, 2))
    j_h = np.ones((1, 1))
    h = np.ones((1, 2))
    ising = Ising("GridQubit", [1, 2], j_v, j_h, h, "X", t)
    bsize=10
    initial_rands= (np.random.rand(bsize, 4)).astype(np.complex128)
    initials = np.zeros(initial_rands.shape, dtype=np.complex64)
    for k in range(bsize):
        initials[k, :] = initial_rands[k, :] / np.linalg.norm(initial_rands[k, :])
    objective = UtCost(ising, t, 0, initial_wavefunctions=initials)
    gd = GradientDescent()
    json = gd.to_json_dict()
    
    gd2 = GradientDescent.from_json_dict(json)
    
    assert gd == gd2

#############################################################
#                     Test errors                           #
#############################################################
def test_GradientDescent_break_cond_assert():
    with pytest.raises(AssertionError):
        GradientDescent({'break_cond': "atol"})