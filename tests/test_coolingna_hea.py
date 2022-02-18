# external imports
import pytest
import numpy as np
import cirq
import sympy

# internal imports
from fauvqe import CoolingNA, Ising

@pytest.mark.parametrize(
    "n, boundaries, options, solution",
    [
        (
            [1, 2], [1, 1],
            {"p": 2},
            list(sympy.symbols('phi0, theta0, psi0, kappa0, mu0, nu0, phi1, theta1, psi1, kappa1, mu1, nu1'))
        ),
        (
            [1, 2], [1, 1],
            {'parametrisation' : 'individual'},
            list(sympy.symbols('theta0_0_1, theta0_0_2, theta0_0_3, theta0_1_2, theta0_1_3, theta0_2_3, phi0_0_1, phi0_0_2, phi0_0_3, phi0_1_2, phi0_1_3, phi0_2_3, psi0_0_1, psi0_0_2, psi0_0_3, psi0_1_2, psi0_1_3, psi0_2_3, kappa0_0_1, kappa0_0_2, kappa0_0_3, kappa0_1_2, kappa0_1_3, kappa0_2_3, mu0_0_1, mu0_0_2, mu0_0_3, mu0_1_2, mu0_1_3, mu0_2_3, nu0_0_1, nu0_0_2, nu0_0_3, nu0_1_2, nu0_1_3, nu0_2_3'))
        )
    ])
def test_set_symbols(n, boundaries, options, solution):
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])))
    m_anc = Ising("GridQubit", n, np.zeros((n[0]-boundaries[0], n[1])), np.zeros((n[0], n[1])), np.ones((n[0], n[1])))
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingNA(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    options.update({
                    "1QubitGates": None,
                    "1Qvariables": [[]],
                    "2Qvariables": [['phi', 'theta'], ['psi', 'kappa'], ['mu', 'nu']],
    })
    model.set_circuit("hea", options)
    assert(set(model.circuit_param) == set(solution)), "Mismatch: {} \n vs. \n {}".format(
        set(model.circuit_param), set(solution)
    )

@pytest.mark.parametrize(
    "n, boundaries, options, solution",
    [
        (
            [1, 2], 
            [1, 1], 
            {"1QubitGates": [lambda x: cirq.XPowGate(exponent=x),
                             lambda z: cirq.ZPowGate(exponent=z)], 
            "1Qvariables": [['x'],['z']],
            "parametrisation": "individual",
            },
            cirq.Circuit((cirq.X**sympy.Symbol('x0_0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('x0_1')).on(cirq.GridQubit(0, 1)),
                         (cirq.X**sympy.Symbol('x0_2')).on(cirq.GridQubit(1, 0)),
                        (cirq.X**sympy.Symbol('x0_3')).on(cirq.GridQubit(1, 1)),
                        (cirq.Z**sympy.Symbol('z0_0')).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**sympy.Symbol('z0_1')).on(cirq.GridQubit(0, 1)),
                        (cirq.Z**sympy.Symbol('z0_2')).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**sympy.Symbol('z0_3')).on(cirq.GridQubit(1, 1))),
        ),
        (
            [1, 2], 
            [1, 1], 
            {"1QubitGates": [lambda x: cirq.XPowGate(exponent=x),
                             lambda z: cirq.ZPowGate(exponent=z)], 
            "1Qvariables": [['x'],['z']],
            "parametrisation": "individual",
            "p": 2
            },
            cirq.Circuit((cirq.X**sympy.Symbol('x0_0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('x0_1')).on(cirq.GridQubit(0, 1)),
                        (cirq.X**sympy.Symbol('x0_2')).on(cirq.GridQubit(1, 0)),
                        (cirq.X**sympy.Symbol('x0_3')).on(cirq.GridQubit(1, 1)),
                        (cirq.Z**sympy.Symbol('z0_0')).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**sympy.Symbol('z0_1')).on(cirq.GridQubit(0, 1)),
                        (cirq.Z**sympy.Symbol('z0_2')).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**sympy.Symbol('z0_3')).on(cirq.GridQubit(1, 1)),
                        (cirq.X**sympy.Symbol('x1_0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('x1_1')).on(cirq.GridQubit(0, 1)),
                        (cirq.X**sympy.Symbol('x1_2')).on(cirq.GridQubit(1, 0)),
                        (cirq.X**sympy.Symbol('x1_3')).on(cirq.GridQubit(1, 1)),
                        (cirq.Z**sympy.Symbol('z1_0')).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**sympy.Symbol('z1_1')).on(cirq.GridQubit(0, 1)),
                        (cirq.Z**sympy.Symbol('z1_2')).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**sympy.Symbol('z1_3')).on(cirq.GridQubit(1, 1))),
        )
    ]
)
def test__1_Qubit_layer(n, boundaries, options, solution):
    m_sys = Ising("GridQubit", n, np.zeros((n[0]-boundaries[0], n[1])), np.zeros((n[0], n[1])), np.ones((n[0], n[1])), field="X")
    m_anc = Ising("GridQubit", n, np.zeros((n[0]-boundaries[0], n[1])), np.zeros((n[0], n[1])), np.ones((n[0], n[1])))
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingNA(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    circuit_options = {"2QubitGates": None}
    circuit_options.update(options)
    model.set_circuit("hea", circuit_options)
    print('Cooling Circuit:\n', model.circuit)
    print('Solution:\n', solution)
    assert model.circuit == solution

@pytest.mark.parametrize(
    "n, boundaries, options, solution",
    [
        (
            [2, 1], 
            [0, 1], 
            {
                "2Qvariables": [['theta', 'phi'], ['kappa', 'psi'], ['mu', 'nu']],
            },
            cirq.Circuit(cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(3, 0)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(2, 0), cirq.GridQubit(3, 0)),
                        cirq.FSimGate(sympy.Symbol('kappa0'), sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.FSimGate(sympy.Symbol('kappa0'), sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        cirq.FSimGate(sympy.Symbol('kappa0'), sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(3, 0)),
                        cirq.FSimGate(sympy.Symbol('kappa0'), sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        cirq.FSimGate(sympy.Symbol('kappa0'), sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
                        cirq.FSimGate(sympy.Symbol('kappa0'), sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(2, 0), cirq.GridQubit(3, 0)),
                         cirq.FSimGate(sympy.Symbol('mu0'), sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.FSimGate(sympy.Symbol('mu0'), sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        cirq.FSimGate(sympy.Symbol('mu0'), sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(3, 0)),
                        cirq.FSimGate(sympy.Symbol('mu0'), sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        cirq.FSimGate(sympy.Symbol('mu0'), sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
                        cirq.FSimGate(sympy.Symbol('mu0'), sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(2, 0), cirq.GridQubit(3, 0)))
        ),
        (
            [1, 2], 
            [1, 1], 
            {
                "2Qvariables": [['theta', 'phi'], ['kappa', 'psi'], ['mu', 'nu']],
            },
            cirq.Circuit(cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
                        cirq.FSimGate(sympy.Symbol('kappa0'), sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.FSimGate(sympy.Symbol('kappa0'), sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.FSimGate(sympy.Symbol('kappa0'), sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)),
                        cirq.FSimGate(sympy.Symbol('kappa0'), sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
                        cirq.FSimGate(sympy.Symbol('kappa0'), sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        cirq.FSimGate(sympy.Symbol('kappa0'), sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
                         cirq.FSimGate(sympy.Symbol('mu0'), sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.FSimGate(sympy.Symbol('mu0'), sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.FSimGate(sympy.Symbol('mu0'), sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)),
                        cirq.FSimGate(sympy.Symbol('mu0'), sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
                        cirq.FSimGate(sympy.Symbol('mu0'), sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        cirq.FSimGate(sympy.Symbol('mu0'), sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)))
        ),
        (
            [1, 2], 
            [1, 1], 
            {"2QubitGates": [lambda phi, theta: cirq.ISwapPowGate (exponent=theta), 
                             lambda psi, kappa: cirq.ISwapPowGate (exponent=kappa)],
            "2Qvariables": [['phi', 'theta'], ['psi', 'kappa']],
            "parametrisation": "individual"},
            cirq.Circuit(cirq.ISwapPowGate (exponent=sympy.Symbol('theta0_0_1')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('theta0_0_2')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('theta0_0_3')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('theta0_1_2')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
                         cirq.ISwapPowGate (exponent=sympy.Symbol('theta0_1_3')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                         cirq.ISwapPowGate (exponent=sympy.Symbol('theta0_2_3')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
                         cirq.ISwapPowGate (exponent=sympy.Symbol('kappa0_0_1')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('kappa0_0_2')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('kappa0_0_3')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('kappa0_1_2')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
                         cirq.ISwapPowGate (exponent=sympy.Symbol('kappa0_1_3')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                         cirq.ISwapPowGate (exponent=sympy.Symbol('kappa0_2_3')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1))),
        )
    ]
)
def test__2_Qubit_layer(n, boundaries, options, solution):
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1])), np.zeros((n[0], n[1])), field="X")
    m_anc = Ising("GridQubit", n, np.zeros((n[0]-boundaries[0], n[1])), np.zeros((n[0], n[1])), np.zeros((n[0], n[1])))
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingNA(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    circuit_options = {"1QubitGates": None}
    circuit_options.update(options)
    model.set_circuit("hea", circuit_options)
    print('Cooling Circuit:\n', model.circuit)
    print('Solution:\n', solution)
    assert model.circuit == solution

@pytest.mark.parametrize(
    "n, boundaries, options, solution",
    [
        (
            [3, 1], 
            [0, 1], 
            {"1QubitGates": [lambda a, x, z: cirq.PhasedXPowGate(exponent=x, phase_exponent=z),
                            lambda a, x, z: cirq.PhasedXPowGate(exponent=x, phase_exponent=z)],
            "2QubitGates": [lambda theta, phi: cirq.ZZPowGate (exponent=theta),
                           lambda mu, nu: cirq.XXPowGate (exponent=mu)],
            "1Qvariables": [['a', 'x', 'z'], ['b', 'y', 'w']],
            "2Qvariables": [['theta', 'phi'], ['mu', 'nu']],
            },
            cirq.Circuit(cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(1, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(2, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(3, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(4, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(5, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('y0'), phase_exponent=sympy.Symbol('w0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('y0'), phase_exponent=sympy.Symbol('w0')).on(cirq.GridQubit(1, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('y0'), phase_exponent=sympy.Symbol('w0')).on(cirq.GridQubit(2, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('y0'), phase_exponent=sympy.Symbol('w0')).on(cirq.GridQubit(3, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('y0'), phase_exponent=sympy.Symbol('w0')).on(cirq.GridQubit(4, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('y0'), phase_exponent=sympy.Symbol('w0')).on(cirq.GridQubit(5, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(3, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(4, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(5, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(1, 0), cirq.GridQubit(4, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(1, 0), cirq.GridQubit(5, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(2, 0), cirq.GridQubit(3, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(2, 0), cirq.GridQubit(4, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(2, 0), cirq.GridQubit(5, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(3, 0), cirq.GridQubit(4, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(3, 0), cirq.GridQubit(5, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0')).on(cirq.GridQubit(4, 0), cirq.GridQubit(5, 0)),
                        
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(3, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(4, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(5, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(1, 0), cirq.GridQubit(4, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(1, 0), cirq.GridQubit(5, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(2, 0), cirq.GridQubit(3, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(2, 0), cirq.GridQubit(4, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(2, 0), cirq.GridQubit(5, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(3, 0), cirq.GridQubit(4, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(3, 0), cirq.GridQubit(5, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0')).on(cirq.GridQubit(4, 0), cirq.GridQubit(5, 0)),
                )
        )
    ]
)
def test_set_circuit(n, boundaries, options, solution):
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1])), np.zeros((n[0], n[1])), field="X")
    m_anc = Ising("GridQubit", n, np.zeros((n[0]-boundaries[0], n[1])), np.zeros((n[0], n[1])), np.zeros((n[0], n[1])))
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingNA(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    model.set_circuit("hea", options)
    #print(ising.circuit)
    assert model.circuit == solution

def test_assert_layerwise():
    n=[1, 2];boundaries=[1, 0]
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1])), np.zeros((n[0], n[1])), field="X")
    m_anc = Ising("GridQubit", n, np.zeros((n[0]-boundaries[0], n[1])), np.zeros((n[0], n[1])), np.zeros((n[0], n[1])))
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingNA(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    with pytest.raises(NotImplementedError):
        model.set_circuit("hea", {'parametrisation': "layerwise"})
    
    model.set_circuit("hea")
    model.hea.options.update({"1QubitGates": None})
    model.hea.options.update({"parametrisation": "layerwise"})
    
    with pytest.raises(NotImplementedError):
        model.hea.set_circuit(model)