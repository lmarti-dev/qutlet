# external imports
import pytest
import numpy as np
import cirq
from scipy.linalg import expm
import sympy

# internal imports
from fauvqe import Ising, Cooling1A

def test__eq__():
    n = [1,3]; boundaries = [1, 0]
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])))
    m_anc = Ising("GridQubit", [1,1], np.zeros((0,1)), np.zeros((1,1)), np.ones((1,1)))
    j_int = np.ones((1, 1))
    
    
    model = Cooling1A(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    model2 = model.copy()
    assert (model == model2)

@pytest.mark.parametrize(
    "qubittype, n, j_x_v, j_x_h, j_y_v, j_y_h, j_z_v, j_z_h, h_x, h_y, h_z",
    [
        #############################################################
        #                   2 qubit tests                           #
        #############################################################
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.zeros((1, 2)) / 10,
            np.zeros((1, 2)) / 10,
            np.zeros((1, 2)) / 10,
        ),
        (
            "GridQubit",
            [2, 1],
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            np.zeros((2, 1)) / 10,
            np.zeros((2, 1)) / 10,
        ),
        (
            "GridQubit",
            [1, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((1, 2)) / 5,
            np.zeros((1, 2)) / 2,
            np.zeros((1, 2)) / 5,
            np.zeros((1, 2)) / 2,
            np.zeros((1, 2)) / 5,
            np.ones((1, 2)) / 3,
            np.ones((1, 2)) / 3,
            np.ones((1, 2)) / 3,
        ),
        #############################################################
        #                   4 qubit tests                           #
        #############################################################
        (
            "GridQubit",
            [2, 2],
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.ones((2, 2)) / 3,
            np.ones((2, 2)) / 3,
            np.ones((2, 2)) / 3,
        ),
        (
            "GridQubit",
            [2, 2],
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.ones((2, 2)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((2, 2)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((2, 2)) / 3,
            np.ones((2, 2)) / 3,
            np.ones((2, 2)) / 3,
        ),
    ],
)
def test_copy(qubittype, n, j_x_v, j_x_h, j_y_v, j_y_h, j_z_v, j_z_h, h_x, h_y, h_z):
    model = Heisenberg(qubittype, n, j_x_v, j_x_h, j_y_v, j_y_h, j_z_v, j_z_h, h_x, h_y, h_z)
    model.set_circuit("qaoa")
    model2 = model.copy()
    
    #Test whether the objects are the same
    assert( model == model2 )
    
    #But there ID is different
    assert( model is not model2 )

def test_json():
    model = Heisenberg("GridQubit", [1, 2], np.zeros((0, 2)), np.ones((1, 1)), np.zeros((0, 2)), np.ones((1, 1)), np.zeros((0, 2)), np.ones((1, 1)), np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2)))
    
    json = model.to_json_dict()
    
    model2 = Heisenberg.from_json_dict(json)
    
    assert (model == model2)

@pytest.mark.parametrize(
    "qubittype, n, j_x_v, j_x_h, j_y_v, j_y_h, j_z_v, j_z_h, h, glue_axis, sol_circuit, sol_circuit_param",
    [
        (
            "GridQubit",
            [1, 3],
            np.ones((0, 3)),
            np.ones((1, 3)),
            np.ones((0, 3)),
            np.ones((1, 3)),
            np.ones((0, 3)),
            np.ones((1, 3)),
            np.ones((1, 3)),
            1,
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1)), cirq.H.on(cirq.GridQubit(0, 2)),
                        cirq.H.on(cirq.GridQubit(0, 3)), cirq.H.on(cirq.GridQubit(0, 4)), cirq.H.on(cirq.GridQubit(0, 5)),
                        (cirq.X**sympy.Symbol('b0_g0')).on(cirq.GridQubit(0, 0)), (cirq.X**sympy.Symbol('b0_g0')).on(cirq.GridQubit(0, 1)),
                        (cirq.X**sympy.Symbol('b0_g0')).on(cirq.GridQubit(0, 2)), (cirq.X**sympy.Symbol('b0_g1')).on(cirq.GridQubit(0, 3)),
                        (cirq.X**sympy.Symbol('b0_g1')).on(cirq.GridQubit(0, 4)), (cirq.X**sympy.Symbol('b0_g1')).on(cirq.GridQubit(0, 5)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g1'))).on(cirq.GridQubit(0, 3), cirq.GridQubit(0, 4)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g0'))).on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 3)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g1'))).on(cirq.GridQubit(0, 5), cirq.GridQubit(0, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g0'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g1'))).on(cirq.GridQubit(0, 4), cirq.GridQubit(0, 5)),),
            [sympy.Symbol('b0_g0'),sympy.Symbol('g0_g0'),sympy.Symbol('b0_g1'),sympy.Symbol('g0_g1')]
        ),
    ]
)
def test_glues_circuit(qubittype, n, j_x_v, j_x_h, j_y_v, j_y_h, j_z_v, j_z_h, h, glue_axis, sol_circuit, sol_circuit_param):
    model = Heisenberg(qubittype, n, j_x_v, j_x_h, j_y_v, j_y_h, j_z_v, j_z_h, h)
    model.set_circuit("qaoa")
    
    model.glue_circuit(axis=glue_axis)
    #print(ising.circuit)

    model2 = Heisenberg(qubittype, 
                    [(2-glue_axis)*n[0], (1+glue_axis)*n[1]], 
                    np.concatenate((j_x_v, j_x_v), axis=glue_axis),
                    np.concatenate((j_x_h, j_x_h), axis=glue_axis) , 
                    np.concatenate((j_y_v, j_y_v), axis=glue_axis),
                    np.concatenate((j_y_h, j_y_h), axis=glue_axis) , 
                     np.concatenate((j_z_v, j_z_v), axis=glue_axis),
                    np.concatenate((j_z_h, j_z_h), axis=glue_axis) , 
                    np.concatenate((h, h), axis=glue_axis) )
    model2.circuit = sol_circuit
    model2.circuit_param = sol_circuit_param
    model2.circuit_param_values = np.array([0]*len(model2.circuit_param))
    
    assert(model == model2)

@pytest.mark.parametrize(
    "qubittype, n, j_x_v, j_x_h, j_y_v, j_y_h, j_z_v, j_z_h, h_x, h_y, h_z",
    [
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2))
        )]
)
def test_energy(qubittype, n, j_x_v, j_x_h, j_y_v, j_y_h, j_z_v, j_z_h, h_x, h_y, h_z):
    model = Heisenberg(qubittype, n, j_x_v, j_x_h, j_y_v, j_y_h, j_z_v, j_z_h, h_x, h_y, h_z)
    obj = ExpectationValue(model)
    ini = np.zeros(16).astype(np.complex64)
    ini[0] = 1
    assert abs(obj.evaluate(ini) + 3) < 1e-13