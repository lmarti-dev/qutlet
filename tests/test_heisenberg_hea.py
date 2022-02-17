# external imports
import pytest
import numpy as np
import cirq
import sympy

# internal imports
from fauvqe import Heisenberg

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
            {"p": 2, "parametrisation": "layerwise"},
            list(sympy.symbols('phi0_1, theta0_1, psi0_1, kappa0_1, mu0_1, nu0_1, phi1_1, theta1_1, psi1_1, kappa1_1, mu1_1, nu1_1'))
        ),
        (
            [1, 2], [1, 1],
            {'parametrisation' : 'individual'},
            list(sympy.symbols('phi0_1_h0_0, theta0_1_h0_0, psi0_1_h0_0, kappa0_1_h0_0, mu0_1_h0_0, nu0_1_h0_0'))
        ),  
    ])
def test_set_symbols(n, boundaries, options, solution):
    #Note that gates & symbols are different based on boundaries
    model = Heisenberg("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])))
    options.update({
                    "1QubitGates": None,
                    "1Qvariables": [[]],
                    "2Qvariables": [['phi', 'theta'], ['psi', 'kappa'], ['mu', 'nu']],
    })
    model.set_circuit("hea", options)
    assert(set(model.circuit_param) == set(solution))

@pytest.mark.parametrize(
    "n, boundaries, options, solution",
    [
        (
            [1, 2], 
            [1, 1], 
            {"1QubitGates": [lambda x: cirq.XPowGate(exponent=x),
                             lambda z: cirq.ZPowGate(exponent=z)], 
            "1Qvariables": [['x'],['z']],
            },
            cirq.Circuit((cirq.X**sympy.Symbol('x0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('x0')).on(cirq.GridQubit(0, 1)),
                        (cirq.Z**sympy.Symbol('z0')).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**sympy.Symbol('z0')).on(cirq.GridQubit(0, 1))),
        ),
        (
            [1, 2], 
            [1, 1], 
            {"1QubitGates": [lambda x: cirq.XPowGate(exponent=x),
                             lambda z: cirq.ZPowGate(exponent=z)], 
            "1Qvariables": [['x'],['z']],
            "p": 2
            },
            cirq.Circuit((cirq.X**sympy.Symbol('x0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('x0')).on(cirq.GridQubit(0, 1)),
                        (cirq.Z**sympy.Symbol('z0')).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**sympy.Symbol('z0')).on(cirq.GridQubit(0, 1)),
                        (cirq.X**sympy.Symbol('x1')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('x1')).on(cirq.GridQubit(0, 1)),
                        (cirq.Z**sympy.Symbol('z1')).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**sympy.Symbol('z1')).on(cirq.GridQubit(0, 1))),
        ),
        (
            [1, 3],
            [1, 1], 
            {
                "1Qvariables": [['a', 'x', 'z'], ['b', 'w', 'y'], ['c', 'u', 'v']],
            },
            cirq.Circuit(cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 1)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 2)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('w0'), 
                                        z_exponent= sympy.Symbol('y0'), 
                                        axis_phase_exponent= sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('w0'), 
                                        z_exponent= sympy.Symbol('y0'), 
                                        axis_phase_exponent= sympy.Symbol('b0')).on(cirq.GridQubit(0, 1)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('w0'), 
                                        z_exponent= sympy.Symbol('y0'), 
                                        axis_phase_exponent= sympy.Symbol('b0')).on(cirq.GridQubit(0, 2)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('u0'), 
                                        z_exponent= sympy.Symbol('v0'), 
                                        axis_phase_exponent= sympy.Symbol('c0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('u0'), 
                                        z_exponent= sympy.Symbol('v0'), 
                                        axis_phase_exponent= sympy.Symbol('c0')).on(cirq.GridQubit(0, 1)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('u0'), 
                                        z_exponent= sympy.Symbol('v0'), 
                                        axis_phase_exponent= sympy.Symbol('c0')).on(cirq.GridQubit(0, 2))),
        ),
        (
            [2, 2], 
            [1, 1], 
            {"1QubitGates": [lambda x: cirq.ry(rads=x), lambda z: cirq.rz(rads=z)], 
            "1Qvariables": [['x'], ['z']],
            "parametrisation": "individual"},
            cirq.Circuit(cirq.ry(sympy.Symbol('x0_0')).on(cirq.GridQubit(0, 0)),
                        cirq.ry(sympy.Symbol('x0_1')).on(cirq.GridQubit(0, 1)),
                        cirq.ry(sympy.Symbol('x0_2')).on(cirq.GridQubit(1, 0)),
                        cirq.ry(sympy.Symbol('x0_3')).on(cirq.GridQubit(1, 1)),
                        cirq.rz(sympy.Symbol('z0_0')).on(cirq.GridQubit(0, 0)),
                        cirq.rz(sympy.Symbol('z0_1')).on(cirq.GridQubit(0, 1)),
                        cirq.rz(sympy.Symbol('z0_2')).on(cirq.GridQubit(1, 0)),
                        cirq.rz(sympy.Symbol('z0_3')).on(cirq.GridQubit(1, 1))),
        )
    ]
)
def test__1_Qubit_layer(n, boundaries, options, solution):
    model = Heisenberg("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])))
    circuit_options = {"2QubitGates": None}
    circuit_options.update(options)
    model.set_circuit("hea", circuit_options)
    print('Heisenberg Circuit:\n', model.circuit)
    print('Solution:\n', solution)
    assert model.circuit == solution

@pytest.mark.parametrize(
    "n, boundaries, options, solution",
    [
        (
            [1, 4], 
            [1, 0], 
            {
                "2Qvariables": [['phi', 'theta'], ['kappa', 'psi'], ['mu', 'nu']],
            },
            cirq.Circuit(cirq.FSimGate(theta=sympy.Symbol('theta0'), phi=sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.FSimGate(theta=sympy.Symbol('theta0'), phi=sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 3)),
                        cirq.FSimGate(theta=sympy.Symbol('theta0'), phi=sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        cirq.FSimGate(theta=sympy.Symbol('theta0'), phi=sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 3), cirq.GridQubit(0, 0)),
                        cirq.FSimGate(phi=sympy.Symbol('kappa0'), theta=sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.FSimGate(phi=sympy.Symbol('kappa0'), theta=sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 3)),
                        cirq.FSimGate(phi=sympy.Symbol('kappa0'), theta=sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        cirq.FSimGate(phi=sympy.Symbol('kappa0'), theta=sympy.Symbol('psi0')).\
                            on(cirq.GridQubit(0, 3), cirq.GridQubit(0, 0)),
                        cirq.FSimGate(phi=sympy.Symbol('mu0'), theta=sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.FSimGate(phi=sympy.Symbol('mu0'), theta=sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 3)),
                        cirq.FSimGate(phi=sympy.Symbol('mu0'), theta=sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        cirq.FSimGate(phi=sympy.Symbol('mu0'), theta=sympy.Symbol('nu0')).\
                            on(cirq.GridQubit(0, 3), cirq.GridQubit(0, 0))),
        ),
        (
            [2, 2], 
            [1, 1], 
            {
                "2QubitGates": [lambda phi, theta: cirq.ZZPowGate(exponent=theta),
                               lambda psi, kappa: cirq.YYPowGate(exponent=kappa)],
                "2Qvariables": [['phi', 'theta'], ['psi', 'kappa']],
                "parametrisation": "layerwise"
            },
            cirq.Circuit(cirq.ZZPowGate(exponent=sympy.Symbol('theta0_0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0_0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0_1')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0_1')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
                        cirq.YYPowGate(exponent=sympy.Symbol('kappa0_0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.YYPowGate(exponent=sympy.Symbol('kappa0_0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        cirq.YYPowGate(exponent=sympy.Symbol('kappa0_1')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.YYPowGate(exponent=sympy.Symbol('kappa0_1')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1))),
        ),
        (
            [2, 2], 
            [1, 1], 
            {"2QubitGates": [lambda phi, theta: cirq.ISwapPowGate (exponent=theta), 
                             lambda psi, kappa: cirq.ISwapPowGate (exponent=kappa)],
             "2Qvariables": [['phi', 'theta'], ['psi', 'kappa']],
            "parametrisation": "individual"},
            cirq.Circuit(cirq.ISwapPowGate (exponent=sympy.Symbol('theta0_0_v0_0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('theta0_0_v0_1')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('theta0_1_h0_0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('theta0_1_h1_0')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('kappa0_0_v0_0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('kappa0_0_v0_1')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('kappa0_1_h0_0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('kappa0_1_h1_0')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1))),
        )
    ]
)
def test__2_Qubit_layer(n, boundaries, options, solution):
    model = Heisenberg("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])))
    circuit_options = {"1QubitGates": None}
    circuit_options.update(options)
    model.set_circuit("hea", circuit_options)
    print('Ising Circuit:\n', model.circuit)
    print('Solution:\n', solution)
    assert model.circuit == solution

@pytest.mark.parametrize(
    "n, boundaries, options, solution",
    [
        (
            [2, 3], 
            [1, 0], 
            {"1QubitGates": [lambda a, x, z: cirq.PhasedXPowGate(exponent=x, phase_exponent=z)],
            "2QubitGates": [lambda theta, phi: cirq.CZPowGate (exponent=theta),
                           lambda kappa, psi: cirq.YYPowGate (exponent=kappa),
                           lambda mu, nu: cirq.XXPowGate (exponent=mu)],
            "1Qvariables": [['a', 'x', 'z']],
            "2Qvariables": [['theta', 'phi'], ['kappa', 'psi'], ['mu', 'nu']],
            "parametrisation": "layerwise"},
            cirq.Circuit(cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(0, 1)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(0, 2)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(1, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(1, 1)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(1, 2)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_0')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_0')).on(cirq.GridQubit(0, 2), cirq.GridQubit(1, 2)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_1')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_1')).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_1')).on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 0)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_1')).on(cirq.GridQubit(1, 2), cirq.GridQubit(1, 0)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_3')).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_3')).on(cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)),
                        cirq.YYPowGate(exponent=sympy.Symbol('kappa0_0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.YYPowGate(exponent=sympy.Symbol('kappa0_0')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        cirq.YYPowGate(exponent=sympy.Symbol('kappa0_0')).on(cirq.GridQubit(0, 2), cirq.GridQubit(1, 2)),
                        cirq.YYPowGate(exponent=sympy.Symbol('kappa0_1')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.YYPowGate(exponent=sympy.Symbol('kappa0_1')).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
                        cirq.YYPowGate(exponent=sympy.Symbol('kappa0_1')).on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 0)),
                        cirq.YYPowGate(exponent=sympy.Symbol('kappa0_1')).on(cirq.GridQubit(1, 2), cirq.GridQubit(1, 0)),
                        cirq.YYPowGate(exponent=sympy.Symbol('kappa0_3')).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        cirq.YYPowGate(exponent=sympy.Symbol('kappa0_3')).on(cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0_0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0_0')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0_0')).on(cirq.GridQubit(0, 2), cirq.GridQubit(1, 2)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0_1')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0_1')).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0_1')).on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0_1')).on(cirq.GridQubit(1, 2), cirq.GridQubit(1, 0)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0_3')).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        cirq.XXPowGate(exponent=sympy.Symbol('mu0_3')).on(cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)),
                )
        )
    ]
)
def test_set_circuit(n, boundaries, options, solution):
    model = Heisenberg("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])))
    model.set_circuit("hea", options)
    #print(ising.circuit)
    assert model.circuit == solution