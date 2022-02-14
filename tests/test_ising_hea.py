# external imports
import pytest
import numpy as np
import cirq
import sympy

# internal imports
from fauvqe import Ising
from .test_isings import IsingTester

"""
What to test:
 (x)     set_symbols()
 ()     _1_Qubit_layer
 ()     _2_Qubit_layer
 ()     set_circuit()
 ()     set_circuit_param_values
 ()     AssertionErros

"""
@pytest.mark.parametrize(
    "n, boundaries, options, solution",
    [
        (
            [1, 2], [1, 1],
            {},
            list(sympy.symbols('a0, phi0, theta0, x0, z0'))
        ), 
        (
            [1, 2], [1, 1],
            {"p": 2},
            list(sympy.symbols('a0, phi0, theta0, x0, z0, a1, phi1, theta1, x1, z1'))
        ),
        (
            [1, 2], [1, 1],
            {"p": 2, "parametrisation": "layerwise"},
            list(sympy.symbols('a0, phi0_1, theta0_1, x0, z0, a1, phi1_1, theta1_1, x1, z1'))
        ),
        (
            [1, 2], [1, 1],
            {'parametrisation' : 'individual'},
            list(sympy.symbols('a0_0, a0_1, x0_0, x0_1, z0_0, z0_1, theta0_1_h0_0, phi0_1_h0_0'))
        ),  
        (
            [2, 2], [0, 0],
            {},
            list(sympy.symbols('a0, phi0, theta0, x0, z0'))
        ), 
        (
            [2, 2], [1, 1],
            {'parametrisation' : 'individual'},
            list(sympy.symbols('a0_0, a0_1, a0_2, a0_3,\
                                x0_0, x0_1, x0_2, x0_3,\
                                z0_0, z0_1, z0_2, z0_3,\
                                theta0_0_v0_0, phi0_0_v0_0,\
                                theta0_0_v0_1, phi0_0_v0_1,\
                                theta0_1_h0_0, phi0_1_h0_0,\
                                theta0_1_h1_0, phi0_1_h1_0' ))
        ),  
        (
            [1, 3], [1, 1],
            {'parametrisation' : 'individual'},
            list(sympy.symbols('a0_0, a0_1, a0_2,\
                                x0_0, x0_1, x0_2,\
                                z0_0, z0_1, z0_2,\
                                theta0_1_h0_0,phi0_1_h0_0,\
                                theta0_3_h0_1,phi0_3_h0_1'))
        ),  
        (
            [1, 3], [1, 0],
            {'parametrisation' : 'individual'},
            list(sympy.symbols('a0_0, a0_1, a0_2\
                                x0_0, x0_1, x0_2\
                                z0_0, z0_1, z0_2,\
                                theta0_1_h0_0, phi0_1_h0_0,\
                                theta0_1_h0_2, phi0_1_h0_2,\
                                theta0_3_h0_1, phi0_3_h0_1'))
        ),  
    ])
def test_set_symbols(n, boundaries, options, solution):
    #Note that gates & symbols are different based on boundaries
    ising_obj = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])))
    ising_obj.set_circuit("hea", options)
    
    assert(set(ising_obj.circuit_param) == set(solution))

@pytest.mark.parametrize(
    "n, boundaries, options, solution",
    [
        (
            [1, 2], 
            [1, 1], 
            {"1QubitGates": [lambda a, x, z: cirq.XPowGate(exponent=x)], 
            "2QubitGates": None 
            },
            cirq.Circuit((cirq.X**sympy.Symbol('x0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('x0')).on(cirq.GridQubit(0, 1))),
        ),
        (
            [1, 2], 
            [1, 1], 
            {"1QubitGates": [lambda a, x, z: cirq.XPowGate(exponent=x)], 
            "2QubitGates": None,
            "p": 2},
            cirq.Circuit((cirq.X**sympy.Symbol('x0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('x0')).on(cirq.GridQubit(0, 1)),
                        (cirq.X**sympy.Symbol('x1')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('x1')).on(cirq.GridQubit(0, 1))),
        ),
        (
            [1, 3], 
            [1, 1], 
            {
                "2QubitGates": None,
            },
            cirq.Circuit(cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 1)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 2))),
        ),
        (
            [1, 2], 
            [1, 1], 
            {"1QubitGates": [lambda a, x, z: cirq.H], 
            "2QubitGates": None, 
            "p": 2,
            },
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)),
                        cirq.H.on(cirq.GridQubit(0, 1)),
                        cirq.H.on(cirq.GridQubit(0, 0)),
                        cirq.H.on(cirq.GridQubit(0, 1))),
        ),
        (
            [2, 2], 
            [1, 1], 
            {"1QubitGates": [lambda x: cirq.ry(rads=x)], 
            "2QubitGates": None, 
            "1Qvariables": 'x',
            "parametrisation": "individual"},
            cirq.Circuit(cirq.ry(sympy.Symbol('x0_0')).on(cirq.GridQubit(0, 0)),
                        cirq.ry(sympy.Symbol('x0_1')).on(cirq.GridQubit(0, 1)),
                        cirq.ry(sympy.Symbol('x0_2')).on(cirq.GridQubit(1, 0)),
                        cirq.ry(sympy.Symbol('x0_3')).on(cirq.GridQubit(1, 1))),
        )
    ]
)
def test__1_Qubit_layer(n, boundaries, options, solution):
    ising = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])))
    circuit_options = {"2QubitGate": None}
    circuit_options.update(options)
    ising.set_circuit("hea", circuit_options)
    print('Ising Circuit:\n', ising.circuit)
    print('Solution:\n', solution)
    assert ising.circuit == solution

@pytest.mark.parametrize(
    "n, boundaries, options, solution",
    [
        (
            [1, 3], 
            [1, 1], 
            {},
            cirq.Circuit(cirq.FSimGate(theta=sympy.Symbol('theta0'), phi=sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.FSimGate(theta=sympy.Symbol('theta0'), phi=sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2))),
        ),
        (
            [1, 3], 
            [1, 0], 
            {},
            cirq.Circuit(cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 2)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2))),
        ),
                (
            [1, 4], 
            [1, 0], 
            {},
            cirq.Circuit(cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 3)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 3), cirq.GridQubit(0, 0))),
        ),
        (
            [2, 2], 
            [1, 1], 
            {
                "2QubitGates": [lambda phi, theta: cirq.ZZPowGate(exponent=theta)],
                "parametrisation": "layerwise"
            },
            cirq.Circuit(cirq.ZZPowGate(exponent=sympy.Symbol('theta0_0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0_0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0_1')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.ZZPowGate(exponent=sympy.Symbol('theta0_1')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1))),
        ),
        (
            [2, 2], 
            [1, 1], 
            {"2QubitGates": [lambda phi, theta: cirq.ISwapPowGate (exponent=theta)],
            "parametrisation": "individual"},
            cirq.Circuit(cirq.ISwapPowGate (exponent=sympy.Symbol('theta0_0_v0_0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('theta0_0_v0_1')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('theta0_1_h0_0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.ISwapPowGate (exponent=sympy.Symbol('theta0_1_h1_0')).\
                            on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1))),
        )
    ]
)
def test__2_Qubit_layer(n, boundaries, options, solution):
    ising = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])))
    circuit_options = {"1QubitGates": None}
    circuit_options.update(options)
    ising.set_circuit("hea", circuit_options)
    print('Ising Circuit:\n', ising.circuit)
    print('Solution:\n', solution)
    assert ising.circuit == solution

@pytest.mark.parametrize(
    "n, boundaries, options, solution",
    [
        (
            [1, 2], 
            [1, 1], 
            {},
            cirq.Circuit(cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 1)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))),
        ),
        (
            [1, 2], 
            [1, 1], 
            {"p": 2},
            cirq.Circuit(cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 1)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x1'), 
                                        z_exponent= sympy.Symbol('z1'), 
                                        axis_phase_exponent= sympy.Symbol('a1')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x1'), 
                                        z_exponent= sympy.Symbol('z1'), 
                                        axis_phase_exponent= sympy.Symbol('a1')).on(cirq.GridQubit(0, 1)),
                        cirq.FSimGate(sympy.Symbol('theta1'), sympy.Symbol('phi1')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))),
        ),
        (
            [1, 3], 
            [1, 1], 
            {},
            cirq.Circuit(cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 1)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 2)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2))),
        ),
        (
            [2, 3], 
            [1, 0], 
            {"1QubitGates": [lambda a, x, z: cirq.PhasedXPowGate(exponent=x, phase_exponent=z)],
            "2QubitGates": [lambda phi, theta: cirq.CZPowGate (exponent=theta)],
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
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_3')).on(cirq.GridQubit(1, 1), cirq.GridQubit(1, 2))),
        )
    ]
)
def test_set_circuit(n, boundaries, options, solution):
    ising = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])))
    ising.set_circuit("hea", options)
    #print(ising.circuit)
    assert ising.circuit == solution

@pytest.mark.parametrize(
    "n, boundaries, options, solution",
    [
        (
            [1, 2], 
            [1, 1], 
            {},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1)), 
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 1)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))),
        ),
        (
            [1, 2], 
            [1, 1], 
            {"p": 2},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1)), 
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 1)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x1'), 
                                        z_exponent= sympy.Symbol('z1'), 
                                        axis_phase_exponent= sympy.Symbol('a1')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x1'), 
                                        z_exponent= sympy.Symbol('z1'), 
                                        axis_phase_exponent= sympy.Symbol('a1')).on(cirq.GridQubit(0, 1)),
                        cirq.FSimGate(sympy.Symbol('theta1'), sympy.Symbol('phi1')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))),
        ),
        (
            [1, 3], 
            [1, 1], 
            {},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1)), cirq.H.on(cirq.GridQubit(0, 2)), 
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 1)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 2)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        cirq.Moment(cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 1)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 2))),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2))),
        ),
        (
            [2, 3], 
            [1, 0], 
            {
                "1QubitGates": [lambda a, x, z: cirq.PhasedXPowGate(exponent=x, phase_exponent=z)],
                "2QubitGates": [lambda phi, theta: cirq.CZPowGate (exponent=theta)],
                "parametrisation": "layerwise"
            },
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1)), cirq.H.on(cirq.GridQubit(0, 2)),
                        cirq.H.on(cirq.GridQubit(1, 0)), cirq.H.on(cirq.GridQubit(1, 1)), cirq.H.on(cirq.GridQubit(1, 2)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)), (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 1)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 2)), (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(1, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(1, 1)), (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(1, 2)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 2), cirq.GridQubit(1, 2)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 2), cirq.GridQubit(1, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)),
                        cirq.Moment(cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(0, 1)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(0, 2)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(1, 0)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(1, 1)),
                        cirq.PhasedXPowGate(exponent=sympy.Symbol('x0'), phase_exponent=sympy.Symbol('z0')).on(cirq.GridQubit(1, 2))),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_0')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_0')).on(cirq.GridQubit(0, 2), cirq.GridQubit(1, 2)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_1')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_1')).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_1')).on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 0)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_1')).on(cirq.GridQubit(1, 2), cirq.GridQubit(1, 0)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_3')).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        cirq.CZPowGate(exponent=sympy.Symbol('theta0_3')).on(cirq.GridQubit(1, 1), cirq.GridQubit(1, 2))),
        )
    ]
)
def test_set_circuit_append(n, boundaries, options, solution):
    ising = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])))
    ising.set_circuit("qaoa")
    options.update({"append":True})
    ising.set_circuit("hea", options)
    print("ising.circuit: \n{}".format(ising.circuit))
    print("solution: \n{}".format(solution))
    assert ising.circuit == solution

@pytest.mark.parametrize(
    "key, values, solution",
    [
        (
            "a", np.array((1,2,3)), np.append([1,2,3],[0]*10)
        )
    ]
)
def test_set_circuit_param_values(key, values, solution):
    n0 = 1;n1 = 3
    b0 = 1; b1 = 1
    ising = Ising("GridQubit",
                [n0, n1],
                np.ones((n0-b0, n1)),
                np.ones((n0, n1-b1)),
                np.ones((n0, n1)),
                "X")

    ising.set_circuit("hea", {"parametrisation": "individual"})
    ising.hea.set_circuit_param_values(ising, key, values)
    assert(ising.circuit_param_values == solution).all()

@pytest.mark.parametrize(
    "options, solution",
    [
        (
            {"parametrisation": "individual",
             "1QubitGates": [lambda x: cirq.XPowGate(exponent=x)], 
             "1Qvariables": [1],
            "2QubitGates": None
            },
            cirq.Circuit((cirq.X.on(cirq.GridQubit(0, 0))),
                        (cirq.X.on(cirq.GridQubit(0, 1)))),
        )
    ]
)
def test_numbers(options, solution):
    n=[1, 2]
    boundaries=[1, 1]
    
    ising = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])))
    circuit_options = {"2QubitGate": None}
    circuit_options.update(options)
    ising.set_circuit("hea", circuit_options)
    print('Ising Circuit:\n', ising.circuit)
    print('Solution:\n', solution)
    assert ising.circuit == solution

def test_erros():
    n0 = 1;n1 = 3
    b0 = 1; b1 = 1
    ising = Ising("GridQubit",
                [n0, n1],
                np.ones((n0-b0, n1)),
                np.ones((n0, n1-b1)),
                np.ones((n0, n1)),
                "X")

    with pytest.raises(AssertionError):
        ising.set_circuit("hea", {'parametrisation': "test"})

    ising.set_circuit("hea")
    ising.hea.options.update({'parametrisation': "test"})

    with pytest.raises(AssertionError):
        ising.hea.set_circuit(ising)
    
    ising.hea.options.update({"1QubitGates": None})
    with pytest.raises(AssertionError):
        ising.hea.set_circuit(ising)
    