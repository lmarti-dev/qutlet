# external imports
import pytest
import numpy as np
import math

# internal imports
from fauvqe import Ising, HeisenbergFC, UtCost

###########################################################
#                       Helper Functions                  #
###########################################################

def fac(q):
    return 1/(4 - 4**(1/(q-1)))


###########################################################
#                       Test Logic                        #
###########################################################

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
            (-2/np.pi)*0.3*np.array([1,2,2]),
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
            0.3,
            1/4*(-2/np.pi)*0.3*np.array([1,2,2,1,2,2,1,2,2,1,2,2]),
        ),
        (
            [2, 1], 
            [1, 1], 
            1,
            2,
            {
                "q":2,
                "m":1,
            },
            1.0,
            0.5*(-2/np.pi)*1.0*np.array([1,2,2,2,2,1]),
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
    params = ising.trotter.get_parameters(ising)
    print(params)
    print(sol)
    assert (abs(params - sol) < 1e-13).all()

@pytest.mark.parametrize(
    "n, J, J2, options, t, sol",
    [
        (
            [2, 1], 
            1,
            2,
            {
                "q":1,
                "m":1,
            },
            0.3,
            (-2/np.pi)*0.3*np.array([1,1,2]),
        ),
        (
            [2, 1], 
            1,
            2,
            {
                "q":1,
                "m":4,
            },
            0.3,
            1/4*(-2/np.pi)*0.3*np.array([1,1,2,1,1,2,1,1,2,1,1,2]),
        ),
        (
            [2, 1], 
            1,
            2,
            {
                "q":2,
                "m":1,
            },
            1.0,
            0.5*(-2/np.pi)*np.array([1,1,2,2,1,1]),
        ),
    ]
)
def test_get_heisenbergfc_trotter_params(n, J, J2, options, t, sol):
    heis = HeisenbergFC("GridQubit", n, 
        J*np.ones((n[0], n[1], n[0], n[1])), 
        J*np.ones((n[0], n[1], n[0], n[1])), 
        J2*np.ones((n[0], n[1], n[0], n[1])),
        t=t
    )
    heis.set_circuit("trotter", options)
    params = heis.trotter.get_parameters(heis)
    print((params))
    print(sol)
    assert (abs(params - sol) < 1e-13).all()


@pytest.mark.parametrize(
    "trotter_opt, prod_opt",
    [
        (
            {
                "q":2,
                "m":1,
            },
            {
                "q":[1],
                "m":1,
            }
        ),
        (
            {
                "q":4,
                "m":1,
            },
            {
                "q":[1-4*fac(4), fac(4), fac(4)],
                "m":1,
            }
        ),
        (
            {
                "q":6,
                "m":1,
            },
            {
                "q":[
                        (1-4*fac(4))*(1-4*fac(6)), fac(4)*(1-4*fac(6)), fac(4)*(1-4*fac(6)), 
                        fac(4)*fac(6), fac(4)*fac(6), (1-4*fac(4))*fac(6), fac(4)*fac(6), fac(4)*fac(6),
                        fac(4)*fac(6), fac(4)*fac(6), (1-4*fac(4))*fac(6), fac(4)*fac(6), fac(4)*fac(6)
                    ],
                "m":1,
            }
        ),
        (
            {
                "q":8,
                "m":1,
            },
            {
                "q":[
                        (1-4*fac(4))*(1-4*fac(6))*(1-4*fac(8)), fac(4)*(1-4*fac(6))*(1-4*fac(8)), fac(4)*(1-4*fac(6))*(1-4*fac(8)), 
                        fac(4)*fac(6)*(1-4*fac(8)), fac(4)*fac(6)*(1-4*fac(8)), (1-4*fac(4))*fac(6)*(1-4*fac(8)), fac(4)*fac(6)*(1-4*fac(8)), fac(4)*fac(6)*(1-4*fac(8)),
                        fac(4)*fac(6)*(1-4*fac(8)), fac(4)*fac(6)*(1-4*fac(8)), (1-4*fac(4))*fac(6)*(1-4*fac(8)), fac(4)*fac(6)*(1-4*fac(8)), fac(4)*fac(6)*(1-4*fac(8)),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                        fac(4)*(1-4*fac(6))*fac(8), fac(4)*(1-4*fac(6))*fac(8), (1-4*fac(4))*(1-4*fac(6))*fac(8), fac(4)*(1-4*fac(6))*fac(8), fac(4)*(1-4*fac(6))*fac(8),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                        fac(4)*(1-4*fac(6))*fac(8), fac(4)*(1-4*fac(6))*fac(8), (1-4*fac(4))*(1-4*fac(6))*fac(8), fac(4)*(1-4*fac(6))*fac(8), fac(4)*(1-4*fac(6))*fac(8),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                    ],
                "m":1,
            }
        ),
        (
            {
                "q":2,
                "m":3,
            },
            {
                "q":[1],
                "m":3,
            }
        ),
        (
            {
                "q":4,
                "m":3,
            },
            {
                "q":[1-4*fac(4), fac(4), fac(4)],
                "m":3,
            }
        ),
        (
            {
                "q":6,
                "m":3,
            },
            {
                "q":[
                        (1-4*fac(4))*(1-4*fac(6)), fac(4)*(1-4*fac(6)), fac(4)*(1-4*fac(6)), 
                        fac(4)*fac(6), fac(4)*fac(6), (1-4*fac(4))*fac(6), fac(4)*fac(6), fac(4)*fac(6),
                        fac(4)*fac(6), fac(4)*fac(6), (1-4*fac(4))*fac(6), fac(4)*fac(6), fac(4)*fac(6)
                    ],
                "m":3,
            }
        ),
        (
            {
                "q":8,
                "m":3,
            },
            {
                "q":[
                        (1-4*fac(4))*(1-4*fac(6))*(1-4*fac(8)), fac(4)*(1-4*fac(6))*(1-4*fac(8)), fac(4)*(1-4*fac(6))*(1-4*fac(8)), 
                        fac(4)*fac(6)*(1-4*fac(8)), fac(4)*fac(6)*(1-4*fac(8)), (1-4*fac(4))*fac(6)*(1-4*fac(8)), fac(4)*fac(6)*(1-4*fac(8)), fac(4)*fac(6)*(1-4*fac(8)),
                        fac(4)*fac(6)*(1-4*fac(8)), fac(4)*fac(6)*(1-4*fac(8)), (1-4*fac(4))*fac(6)*(1-4*fac(8)), fac(4)*fac(6)*(1-4*fac(8)), fac(4)*fac(6)*(1-4*fac(8)),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                        fac(4)*(1-4*fac(6))*fac(8), fac(4)*(1-4*fac(6))*fac(8), (1-4*fac(4))*(1-4*fac(6))*fac(8), fac(4)*(1-4*fac(6))*fac(8), fac(4)*(1-4*fac(6))*fac(8),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                        fac(4)*(1-4*fac(6))*fac(8), fac(4)*(1-4*fac(6))*fac(8), (1-4*fac(4))*(1-4*fac(6))*fac(8), fac(4)*(1-4*fac(6))*fac(8), fac(4)*(1-4*fac(6))*fac(8),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                        fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8), (1-4*fac(4))*fac(6)*fac(8), fac(4)*fac(6)*fac(8), fac(4)*fac(6)*fac(8),
                    ],
                "m":3,
            }
        ),
    ]
)
def test_product_formulas(trotter_opt, prod_opt):
    n=[2, 1]
    boundaries = [1, 1]
    J=1
    h=2
    t=1.0
    ising = Ising("GridQubit", n, 
        J*np.ones((n[0]-boundaries[0], n[1])), 
        J*np.ones((n[0], n[1]-boundaries[1])), 
        h*np.ones((n[0], n[1])),
        t=t
    )
    ising.set_circuit("trotter", trotter_opt)
    ising2 = Ising("GridQubit", n, 
        J*np.ones((n[0]-boundaries[0], n[1])), 
        J*np.ones((n[0], n[1]-boundaries[1])), 
        h*np.ones((n[0], n[1])),
        t=t
    )
    ising2.set_circuit("trotter", prod_opt)
    print(len(ising.circuit))
    print(len(ising2.circuit))
    assert len(ising.circuit) == len(ising2.circuit), "Circuits do not coincide... \nCircuit 1:\n{}\nCircuit 2:\n{}".format(ising.circuit, ising2.circuit)
    assert np.linalg.norm(ising.circuit.unitary() - ising2.circuit.unitary()) < 1e-7, "Unitaries does not coincide... \nUnitary 1:\n{}\nUnitary 2:\n{}".format(ising.circuit.unitary(), ising2.circuit.unitary())


###########################################################
#                        Test Errors                      #
###########################################################

@pytest.mark.parametrize(
    "n, boundaries, J, h, options, t",
    [
        (
            [2, 1], 
            [1, 1], 
            1,
            2,
            {
                "q":4,
                "m":1,
            },
            1.0,
        ),
    ]
)
def test_get_params_errors(n, boundaries, J, h, options, t):
    ising = Ising("GridQubit", n, 
        J*np.ones((n[0]-boundaries[0], n[1])), 
        J*np.ones((n[0], n[1]-boundaries[1])), 
        h*np.ones((n[0], n[1])),
        t=t)
    ising.set_circuit("trotter", options)
    with pytest.raises(NotImplementedError):
        ising.trotter.get_parameters(ising)