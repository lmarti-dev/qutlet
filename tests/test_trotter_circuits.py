# external imports
import pytest
import numpy as np
import scipy

# internal imports
from fauvqe import Ising, HeisenbergFC, UtCost

@pytest.mark.parametrize(
    "n, boundaries, field, options, t",
    [
        (
            [2, 1], 
            [1, 1], 
            "X",
            {
                "q":1,
                "m":1,
            },
            0.3
        ),
        (
            [2, 1], 
            [1, 1], 
            "Z",
            {
                "q":1,
                "m":1,
            },
            0.3
        ),
        (
            [2, 1], 
            [1, 1], 
            "X",
            {
                "q":4,
                "m":1,
            },
            0.3
        ),
        (
            [2, 1], 
            [1, 1], 
            "X",
            {
                "q":1,
                "m":4,
            },
            1.0
        ),
    ]
)
def test_set_ising_trotter(n, boundaries, field, options, t):
    ising = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])), field, t=t)
    ising.set_circuit("trotter", options)
    ising.set_simulator("cirq", dtype=np.complex128)
    cost = UtCost(ising, t)
    
    assert cost.evaluate(cost.simulate({})) < 1e-2

@pytest.mark.parametrize(
    "n, options, t",
    [
        (
            [2, 1], 
            {
                "q":1,
                "m":1,
            },
            0.3
        ),
        (
            [2, 1], 
            {
                "q":1,
                "m":1,
            },
            0.3
        ),
        (
            [2, 1], 
            {
                "q":4,
                "m":1,
            },
            0.3
        ),
        (
            [2, 1], 
            {
                "q":1,
                "m":4,
            },
            1.0
        ),
    ]
)
def test_set_heisenbergfc_trotter(n, options, t):
    heis = HeisenbergFC("GridQubit", n, np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1], n[0], n[1])), t=t)
    heis.set_circuit("trotter", options)
    heis.set_simulator("cirq", dtype=np.complex128)
    cost = UtCost(heis, t)
    
    assert cost.evaluate(cost.simulate({})) < 1e-2

@pytest.mark.parametrize(
    "n, boundaries, J, h, options, t, sol",
    [
        (
            [2, 1], 
            [1, 1], 
            1,
            2,
            {
                "q":1,
                "m":1,
            },
            0.3,
            (2/np.pi)*0.3*np.array([1,2,2]),
        ),
        (
            [2, 1], 
            [1, 1], 
            1,
            2,
            {
                "q":1,
                "m":1,
            },
            0.3,
            (2/np.pi)*0.3*np.array([1,2,2]),
        ),
        (
            [2, 1], 
            [1, 1], 
            1,
            2,
            {
                "q":4,
                "m":1,
            },
            0.3,
            (2/np.pi)*0.3*np.array([1,2,2]),
        ),
        (
            [2, 1], 
            [1, 1], 
            1,
            2,
            {
                "q":1,
                "m":4,
            },
            1.0,
            (2/np.pi)*0.3*np.array([1,2,2]),
        ),
    ]
)
def test_get_ising_trotter_params(n, boundaries, J, h, options, t, sol):
    ising = Ising("GridQubit", n, 
        J*np.ones((n[0]-boundaries[0], n[1])), 
        J*np.ones((n[0], n[1]-boundaries[1])), 
        h*np.ones((n[0], n[1])),
        t=t)
    ising.set_circuit("trotter", options)
    
    assert ((ising.trotter.get_parameters() - sol) < 1e-13).all()

@pytest.mark.parametrize(
    "n, options, t",
    [
        (
            [2, 1], 
            {
                "q":1,
                "m":1,
            },
            0.3
        ),
        (
            [2, 1], 
            {
                "q":1,
                "m":1,
            },
            0.3
        ),
        (
            [2, 1], 
            {
                "q":4,
                "m":1,
            },
            0.3
        ),
        (
            [2, 1], 
            {
                "q":1,
                "m":4,
            },
            1.0
        ),
    ]
)
def test_get_heisenbergfc_trotter_params(n, options, t):
    heis = HeisenbergFC("GridQubit", n, np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1], n[0], n[1])), t=t)
    heis.set_circuit("trotter", options)
    heis.set_simulator("cirq", dtype=np.complex128)
    cost = UtCost(heis, t)
    
    assert cost.evaluate(cost.simulate({})) < 1e-2