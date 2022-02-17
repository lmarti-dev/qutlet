# external imports
import pytest
import numpy as np
import cirq

from numbers import Real, Integral
from typing import Literal, Optional, Dict, List
# internal imports
from fauvqe import Ising, ExpectationValue, UtCost, GradientOptimiser, Optimiser

class MockGradientOptimiser(GradientOptimiser):
    def _optimise_step(self, temp_cpv: np.ndarray, _n_jobs: Integral, step: Integral):
        super()._optimise_step(temp_cpv, _n_jobs, step)

@pytest.mark.parametrize(
    "sym",
    [
        (True),(False)
    ],
)
@pytest.mark.higheffort
def test_get_single_energy(sym):
    ising = Ising(
        "GridQubit", [2, 2], 0.1 * np.ones((1, 2)), 0.5 * np.ones((2, 1)), 0.2 * np.ones((2, 2))
    )
    ising.set_circuit("qaoa", {"p": 2})
    ising.set_circuit_param_values(0.3 * np.ones(np.size(ising.circuit_param)))
    
    mockopt = MockGradientOptimiser({
        'break_param': 1,
        'eta': 4e-2,
        'symmetric_gradient': sym,     
    })
    expval_z = ExpectationValue(ising)
    mockopt._objective = expval_z
    mockopt._circuit_param = ising.circuit_param
    mockopt._n_param = np.size(ising.circuit_param)
    gg_gradients, cost = mockopt._get_gradients(mockopt._objective.model.circuit_param_values, 8)
    if(sym):
        reps = 2*mockopt._n_param
    else:
        reps = mockopt._n_param + 1
    single_energies = np.zeros(reps)
    for j in range(reps):
        single_energies[j] = mockopt._get_single_cost(
            {**{str(mockopt._circuit_param[i]): mockopt._objective.model.circuit_param_values[i] for i in range(mockopt._n_param)}} , 
            j)
    if(sym):
        single_energies = np.array(single_energies).reshape((mockopt._n_param, 2)) 
        se_gradients = np.matmul(single_energies, np.array((1, -1))) / (2 * mockopt.options['eps']) 
    else:
        se_gradients = (single_energies[1:] - single_energies[0]) / (mockopt.options['eps'])
    np.testing.assert_allclose(gg_gradients, se_gradients, rtol=1e-15, atol=1e-15)

@pytest.mark.parametrize(
    "sym",
    [
        (True),(False)
    ],
)
@pytest.mark.higheffort
def test__get_single_energy_batch(sym):
    ising = Ising(
        "GridQubit", [2, 2], 0.1 * np.ones((1, 2)), 0.5 * np.ones((2, 1)), 0.2 * np.ones((2, 2))
    )
    ising.set_circuit("qaoa", {"p": 2})
    ising.set_circuit_param_values(0.3 * np.ones(np.size(ising.circuit_param)))
    
    mockopt = MockGradientOptimiser({
        'break_param': 1,
        'eta': 4e-2,
        'symmetric_gradient': sym,
        'batch_size': 1
    })
    obj = UtCost(ising, 0.1, 0, initial_wavefunctions=np.array([np.random.rand(16).astype(np.complex64)]))
    mockopt._objective = obj
    mockopt._circuit_param = ising.circuit_param
    mockopt._n_param = np.size(ising.circuit_param)
    gg_gradients, cost = mockopt._get_gradients(mockopt._objective.model.circuit_param_values, 8, indices=[0])
    if(sym):
        reps = 2*mockopt._n_param
    else:
        reps = mockopt._n_param + 1
    single_energies = np.zeros(reps)
    for j in range(reps):
        single_energies[j] = mockopt._get_single_cost(
            {**{str(mockopt._circuit_param[i]): mockopt._objective.model.circuit_param_values[i] for i in range(mockopt._n_param)}} , 
            j, [0])
    if(sym):
        single_energies = np.array(single_energies).reshape((mockopt._n_param, 2)) 
        se_gradients = np.matmul(single_energies, np.array((1, -1))) / (2 * mockopt.options['eps']) 
    else:
        se_gradients = (single_energies[1:] - single_energies[0]) / (mockopt.options['eps'])
    np.testing.assert_allclose(gg_gradients, se_gradients, rtol=1e-15, atol=1e-15)

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
    mockopt = MockGradientOptimiser()
    json = mockopt.to_json_dict()
    
    mockopt2 = MockGradientOptimiser.from_json_dict(json)
    
    assert mockopt.__eq__(mockopt2)

class MockGradientOptimiser(GradientOptimiser):
    def _optimise_step(self, temp_cpv: np.ndarray, _n_jobs: Integral, step: Integral):
        super()._optimise_step(temp_cpv, _n_jobs, step)

        
#############################################################
#                                                           #
#                     Test errors                           #
#                                                           #
#############################################################
def test_abstract_gradient_optimiser():
    with pytest.raises(TypeError):
        GradientOptimiser()

def test_optimiser_optimise_assert():
    # Test if abstract base class can be initiated
    with pytest.raises(TypeError):
        Optimiser()
        
def test_parameter_update_NotImplemented():
    with pytest.raises(NotImplementedError):
        mockopt = MockGradientOptimiser()
        mockopt._optimise_step(np.ones(2), 1, 1)