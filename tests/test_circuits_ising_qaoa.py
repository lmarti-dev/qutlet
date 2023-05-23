# external imports
import pytest
import numpy as np
import cirq
import sympy

# internal imports
from fauvqe import Ising
from .test_models_isings import IsingTester

"""
What to test:

 (x)   self.set_circuit(qalgorithm, param, append)
        Assertion Error if qalgorithm != 'qaoa'
                        'qaoa', param higher dimmension

(1) self.qaoa.set_circuit # This is not the same as self.set_circuit()!!
        -> test together with ising_ob.set_circuit()
(x) self.qaoa.set_symbols


(x) self.qaoa.set_p
(1) self.qaoa.set_beta_values
(1) self.qaoa.set_gamma_values

(x) self.qaoa._UB_layer
(x) self.qaoa._UC_layer
(x) self.qaoa._get_param_resolver
"""


def test_set_circuit_wf():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_circuit("qaoa")
    ising.set_simulator("cirq", {"dtype": np.complex128})
    print(ising.circuit)
    assert ising.qaoa.options["p"]== 1
    assert ising.circuit_param == [sympy.Symbol("b0"), sympy.Symbol("g0")]
    # Basic test whether circuit gives correct result
    temp_cpv = np.array((1, 0))
    joined_dict = {
        **{
            str(ising.circuit_param[i]): temp_cpv[i]
            for i in range(np.size(ising.circuit_param_values))
        }
    }
    obj_param_resolver = cirq.ParamResolver(joined_dict)

    wf_x = ising.simulator.simulate(
        ising.circuit, param_resolver=obj_param_resolver
    ).state_vector()
    #Normalise wavevector
    wf_x = wf_x/np.linalg.norm(wf_x)
    assert np.allclose(
        wf_x,
        np.array([0.5, 0.5,0.5,0.5]),
        rtol=0,
        atol=1e-14,
    )

    ising.field="Z"
    ising.set_circuit("qaoa", {"append": False})
    print(ising.circuit)
    temp_cpv = np.array((1, 0.5))
    joined_dict = {
        **{
            str(ising.circuit_param[i]): temp_cpv[i]
            for i in range(np.size(ising.circuit_param_values))
        }
    }
    obj_param_resolver = cirq.ParamResolver(joined_dict)

    wf_z = np.array(ising.simulator.simulate(
        ising.circuit, param_resolver=obj_param_resolver
    ).state_vector())
    #Normalise wavevector
    wf_z = wf_z/np.linalg.norm(wf_z)

    assert np.allclose(
        wf_z,
        np.array((-0.5, -0.5, -0.5, 0.5), dtype=complex),
        rtol=0,
        atol=1e-14,
    )

@pytest.mark.parametrize(
    "n, boundaries, field, options, solution",
    [
        (
            [1, 2], 
            [1, 1], 
            "X",
            {},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1)), 
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0)), (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 1)),
            )
                        
        ),
        (
            [1, 3], 
            [1, 1], 
            "X",
            {},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1)), cirq.H.on(cirq.GridQubit(0, 2)), 
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0)), 
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 2)),
                        cirq.Moment((cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 1)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 2))),
            )
        ),
         (
            [1, 3], 
            [1, 0], 
            "X",
            {},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1)), cirq.H.on(cirq.GridQubit(0, 2)), 
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0)), (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 2)),
                        cirq.Moment((cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 1)),
                       (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 2))),
            )
                        
        ),
        (
            [1, 3], 
            [1, 0], 
            "X",
            {"p":2},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1)), cirq.H.on(cirq.GridQubit(0, 2)), 
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0)), 
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1)),
                        (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 2)),
                        cirq.Moment((cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                                    (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 1)),
                                    (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 2))),
                        (cirq.ZZ**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        (cirq.Z**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(0, 0)), 
                        (cirq.Z**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(0, 1)),
                        (cirq.Z**(1.0*sympy.Symbol('g1'))).on(cirq.GridQubit(0, 2)), 
                        cirq.Moment((cirq.X**sympy.Symbol('b1')).on(cirq.GridQubit(0, 0)),
                                    (cirq.X**sympy.Symbol('b1')).on(cirq.GridQubit(0, 1)),
                                    (cirq.X**sympy.Symbol('b1')).on(cirq.GridQubit(0, 2))),
                        ),
        ),
        (
            [3, 1], 
            [0, 1], 
            "Z",
            {},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)), cirq.H.on(cirq.GridQubit(2, 0)),  
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 0), cirq.GridQubit(0, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        cirq.Moment((cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0)),
                                    (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0)),
                                    (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 0))),
                        cirq.Moment((cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                                    (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(1, 0)),
                                    (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(2, 0))),
            ),            
        ),
        (
            [3, 3], 
            [0, 0], 
            "Z",
            {},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)), cirq.H.on(cirq.GridQubit(2, 0)), 
                        cirq.H.on(cirq.GridQubit(0, 1)), cirq.H.on(cirq.GridQubit(1, 1)), cirq.H.on(cirq.GridQubit(2, 1)), 
                        cirq.H.on(cirq.GridQubit(0, 2)), cirq.H.on(cirq.GridQubit(1, 2)), cirq.H.on(cirq.GridQubit(2, 2)), 
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 2), cirq.GridQubit(1, 2)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 0), cirq.GridQubit(0, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 1), cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 2), cirq.GridQubit(0, 2)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 0), cirq.GridQubit(2, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 2), cirq.GridQubit(1, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 2), cirq.GridQubit(2, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 1), cirq.GridQubit(2, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 2), cirq.GridQubit(2, 2)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 1), cirq.GridQubit(2, 2)),
                        cirq.Moment((cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0)),
                                    (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1)),
                                    (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 2)),
                                    (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 0)),
                                    (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 1)),
                                    (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(1, 2)),
                                    (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 0)),
                                    (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 1)),
                                    (cirq.Z**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(2, 2))),
                        cirq.Moment((cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                                    (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 1)),
                                    (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 2)),
                                    (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(1, 0)),
                                    (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(1, 1)),
                                    (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(1, 2)),
                                    (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(2, 0)),
                                    (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(2, 1)),
                                    (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(2, 2))),
            ),
                        
        ),
    ]
)
def test_set_circuit(n, boundaries, field, options, solution):
    ising = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])), field)
    ising.set_circuit("qaoa", options)
    print("ising.circuit: \n{}".format(ising.circuit))
    print("solution: \n{}".format(solution))
    assert ising.circuit == solution

# still needs to be improved
def test_set_p():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_circuit("qaoa")
    assert ising.qaoa.options["p"] == 1
    ising.qaoa.set_p(ising, 2)
    assert ising.qaoa.options["p"] == 2


# test more examples
# test assertion error
def test_set_beta_values():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_circuit("qaoa", {"p":2})
    ising.qaoa._set_beta_values(ising, [0.2, 0.3])
    # Assert whether self.circuit_param_values has been set correctly
    assert (ising.circuit_param_values == np.array([0.2, 0.0, 0.3, 0.0])).all()


# test more examples
# test assertion error
def test_set_gamma_values():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_circuit("qaoa", {"p": 2})
    ising.qaoa._set_gamma_values(ising, [0.4, 0.5])
    # Assert whether self.circuit_param_values has been set correctly
    assert (ising.circuit_param_values == np.array([0.0, 0.4, 0.0, 0.5])).all()


def test__UB_layer():
    self = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    self.circuit_param_values = np.array([1, 0.0])
    # Watchout replaced here circuit_param with circuit_param_value
    # This distroys the circuit parametrisation, just do for testing
    self.circuit.append(cirq.Moment(self.qaoa._UB_layer(self, self.circuit_param_values[0])))
    assert np.allclose(
        self.circuit.unitary(),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        rtol=0,
        atol=1e-14,
    )


def test__UC_layer():
    self = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    self.circuit_param_values = np.array([0.0, 1.0])
    # Watchout replaced here circuit_param with circuit_param_value
    # This distroys the circuit parametrisation, just do for testing
    self.circuit.append(self.qaoa._UC_layer(self, self.circuit_param_values[1]))
    assert np.allclose(
        self.circuit.unitary(),
        np.array(
            [
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            ]
        ),
        rtol=0,
        atol=1e-14,
    )
    # Cover case n[0] > 1
    self = Ising("GridQubit", [2, 1], np.ones((1, 1)), np.ones((2, 0)), np.ones((2, 1)))
    self.circuit_param_values = np.array([0.0, 1.0])
    self.circuit.append(self.qaoa._UC_layer(self, self.circuit_param_values[1]))
    assert np.allclose(
        self.circuit.unitary(),
        np.array(
            [
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            ]
        ),
        rtol=0,
        atol=1e-14,
    )

@pytest.mark.parametrize('p', [(1), (2), (3)])
def test_get_param_resolver(p):
    self = Ising("GridQubit", [2, 1], np.ones((1, 1)), np.ones((2, 0)), np.ones((2, 1)))
    self.qaoa.options["p"] = p

    b_values = np.random.rand(p)
    g_values = np.random.rand(p)
    param_resolver = self.qaoa._get_param_resolver(self, b_values, g_values)
    if p == 1:
        assert(param_resolver == cirq.ParamResolver(
            {**{"b0": b_values}, **{"g0": g_values}} ))
    else:
        assert(param_resolver == cirq.ParamResolver(
            {**{"b" + str(i): b_values[i] for i in range(p)},
            **{"g" + str(i): g_values[i] for i in range(p)},}))


def test_set_circuit_erros():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    with pytest.raises(AssertionError):
        ising.set_circuit("test")

def test_set_p_errors():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.qaoa.set_p(ising, 2)
    assert ising.qaoa.options["p"] == 2

def test_set_beta_values_errors():
    self = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    self.set_circuit("qaoa")
    with pytest.raises(AssertionError):
        self.qaoa._set_beta_values(self, np.array((0, 1)))


def test_set_gamma_values_errors():
    self = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    self.set_circuit("qaoa")
    with pytest.raises(AssertionError):
        self.qaoa._set_gamma_values(self, np.array((0, 1)))

def test_i0():
    self = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    
    #Set some parametrised circuit 
    self.set_circuit("hea")
    print(self.circuit)
    self.set_circuit("qaoa", {"append": True})
    assert np.size(self.circuit_param) == np.size(self.circuit_param_values)
    
    #Test whether set beta/gamma values works with i0
    self.qaoa._set_beta_values(self, 1)
    temp = np.zeros(np.shape(self.circuit_param_values))
    temp[-2] = 1
    assert (self.circuit_param_values == temp).all()

    self.qaoa._set_gamma_values(self, 1)
    temp[-1] = 1
    assert (self.circuit_param_values == temp).all()

    #Test whether set beta/gamma values errors works with i0
    self.circuit_param_values = np.zeros(np.size(self.circuit_param_values)-1)
    with pytest.raises(AssertionError):
        self.qaoa._set_beta_values(self, 0)
    
    with pytest.raises(AssertionError):
        self.qaoa._set_gamma_values(self, 0)