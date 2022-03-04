# external imports
import pytest
import numpy as np
import cirq
import sympy

# internal imports
from fauvqe import Ising, CoolingModel

def test_set_circuit_wf():
    n=[1,2]
    m_sys = Ising("GridQubit", n, np.ones((n[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])))
    m_anc = Ising("GridQubit", n, np.zeros((n[0], n[1])), np.zeros((n[0], n[1])), np.ones((n[0], n[1])))
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingModel(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    model.set_circuit("qaoa")
    print(model.circuit)
    assert model.qaoa.options["p"]== 1
    assert model.circuit_param == [sympy.Symbol("b0"), sympy.Symbol("g0")]
    # Basic test whether circuit gives correct result
    temp_cpv = np.array((1, 0))
    joined_dict = {
        **{
            str(model.circuit_param[i]): temp_cpv[i]
            for i in range(np.size(model.circuit_param_values))
        }
    }
    obj_param_resolver = cirq.ParamResolver(joined_dict)

    wf_x = model.simulator.simulate(
        model.circuit, param_resolver=obj_param_resolver
    ).state_vector()
    #Normalise wavevector
    wf_x = wf_x/np.linalg.norm(wf_x)
    assert np.allclose(
        wf_x,
        0.25*np.ones(16),
        rtol=0,
        atol=1e-14,
    )

@pytest.mark.parametrize(
    "n, boundaries, field, options, solution",
    [
        (
            [2, 1], 
            [1, 1], 
            "X",
            {"1QubitGates": [lambda q, theta: cirq.X(q)**(theta),
                             lambda q, theta: cirq.Z(q)**(theta)],
             "2QubitGates" : [lambda q1, q2, theta: cirq.ZZ(q1, q2)**(theta),
                              lambda q1, q2, theta: cirq.ZZ(q1, q2)**(theta),
                             lambda q1, q2, theta: cirq.XX(q1, q2)**(theta)]},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)), cirq.H.on(cirq.GridQubit(2, 0)), cirq.H.on(cirq.GridQubit(3, 0)), 
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(1, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(2, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(3, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
                        (cirq.X**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(3, 0))),
        ),
        (
            [3, 1], 
            [1, 1], 
            "X",
            {"1QubitGates": [lambda q, theta: cirq.Z(q)**(theta),
                             lambda q, theta: cirq.Z(q)**(theta)],
             "2QubitGates" : [lambda q1, q2, theta: cirq.ZZ(q1, q2)**(theta),
                              lambda q1, q2, theta: cirq.ZZ(q1, q2)**(theta),
                             lambda q1, q2, theta: cirq.XX(q1, q2)**(theta)]},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)), cirq.H.on(cirq.GridQubit(2, 0)),cirq.H.on(cirq.GridQubit(3, 0)), cirq.H.on(cirq.GridQubit(4, 0)), cirq.H.on(cirq.GridQubit(5, 0)), 
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(1, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(2, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(3, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(4, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(5, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(3, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0), cirq.GridQubit(4, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 0), cirq.GridQubit(5, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(3, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(4, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(5, 0))),
        ),
        (
            [3, 1], 
            [0,1], 
            "X",
            {"1QubitGates": [lambda q, theta: cirq.Z(q)**(theta),
                             lambda q, theta: cirq.Z(q)**(theta)],
             "2QubitGates" : [lambda q1, q2, theta: cirq.ZZ(q1, q2)**(theta),
                              lambda q1, q2, theta: cirq.ZZ(q1, q2)**(theta),
                             lambda q1, q2, theta: cirq.XX(q1, q2)**(theta)]},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)), cirq.H.on(cirq.GridQubit(2, 0)),cirq.H.on(cirq.GridQubit(3, 0)), cirq.H.on(cirq.GridQubit(4, 0)), cirq.H.on(cirq.GridQubit(5, 0)), 
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(1, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(2, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(3, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(4, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(5, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 0), cirq.GridQubit(0, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(3, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0), cirq.GridQubit(4, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 0), cirq.GridQubit(5, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(3, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(4, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(5, 0))),
        ),
        (
            [3, 1], 
            [0, 1], 
            "X",
            {"p":2,
             "1QubitGates": [lambda q, theta: cirq.Z(q)**(theta),
                             lambda q, theta: cirq.Z(q)**(theta)],
            "2QubitGates" : [lambda q1, q2, theta: cirq.ZZ(q1, q2)**(theta),
                              lambda q1, q2, theta: cirq.ZZ(q1, q2)**(theta),
                             lambda q1, q2, theta: cirq.XX(q1, q2)**(theta)]},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)), cirq.H.on(cirq.GridQubit(2, 0)),cirq.H.on(cirq.GridQubit(3, 0)), cirq.H.on(cirq.GridQubit(4, 0)), cirq.H.on(cirq.GridQubit(5, 0)), 
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(1, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(2, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(3, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(4, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(5, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 0), cirq.GridQubit(0, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(3, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0), cirq.GridQubit(4, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 0), cirq.GridQubit(5, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(3, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(4, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(5, 0)),
                        (cirq.X**sympy.Symbol('b1')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('b1')).on(cirq.GridQubit(1, 0)),
                        (cirq.X**sympy.Symbol('b1')).on(cirq.GridQubit(2, 0)),
                        (cirq.X**sympy.Symbol('b1')).on(cirq.GridQubit(3, 0)),
                        (cirq.X**sympy.Symbol('b1')).on(cirq.GridQubit(4, 0)),
                        (cirq.X**sympy.Symbol('b1')).on(cirq.GridQubit(5, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(2, 0), cirq.GridQubit(0, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(3, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(1, 0), cirq.GridQubit(4, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(2, 0), cirq.GridQubit(5, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(2, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(3, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(4, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(5, 0))),
        ),
        (
            [1, 3], 
            [1, 0], 
            "Z",
            {"1QubitGates": [lambda q, theta: cirq.Z(q)**(theta),
                             lambda q, theta: cirq.Z(q)**(theta)],
             "2QubitGates": [lambda q1, q2, theta: cirq.ZZ(q1, q2)**(theta),
                              lambda q1, q2, theta: cirq.ZZ(q1, q2)**(theta),
                             lambda q1, q2, theta: cirq.XX(q1, q2)**(theta)]},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1)), cirq.H.on(cirq.GridQubit(0, 2)),cirq.H.on(cirq.GridQubit(1, 0)), cirq.H.on(cirq.GridQubit(1, 1)), cirq.H.on(cirq.GridQubit(1, 2)), 
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 1)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 2)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(1, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(1, 1)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(1, 2)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        (cirq.XX**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        (cirq.XX**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 2), cirq.GridQubit(1, 2)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 2)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 1)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 2))),
        ),
        
    ]
)
def test_set_circuit(n, boundaries, field, options, solution):
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])), field)
    m_anc = Ising("GridQubit", n, np.zeros((n[0], n[1])), np.zeros((n[0], n[1])), np.ones((n[0], n[1])))
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingModel(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    model.set_circuit("qaoa", options)
    print("model.circuit: \n{}".format(cirq.align_left(model.circuit)))
    print("solution: \n{}".format(cirq.align_left(solution)))
    assert (cirq.align_left(model.circuit)) == (cirq.align_left(solution))


@pytest.mark.parametrize('p', [(1), (2), (3)])
def test_get_param_resolver(p):
    n=[1,2];boundaries=[1, 0]
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])))
    m_anc = Ising("GridQubit", n, np.zeros((n[0], n[1])), np.zeros((n[0], n[1])), np.ones((n[0], n[1])))
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingModel(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    model.qaoa.options["p"] = p

    b_values = np.random.rand(p)
    g_values = np.random.rand(p)
    param_resolver = model.qaoa._get_param_resolver(model, b_values, g_values)
    if p == 1:
        assert(param_resolver == cirq.ParamResolver(
            {**{"b0": b_values}, **{"g0": g_values}} ))
    else:
        assert(param_resolver == cirq.ParamResolver(
            {**{"b" + str(i): b_values[i] for i in range(p)},
            **{"g" + str(i): g_values[i] for i in range(p)},}))