# external imports
import pytest
import numpy as np
import cirq
import sympy

# internal imports
from fauvqe import Ising
from ..tests.ising_tests import IsingTester

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


def test_set_circuit():
    ising_obj = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising_obj.set_circuit("qaoa", 1)
    print(ising_obj.circuit)
    assert ising_obj.p == 1
    assert ising_obj.circuit_param == [sympy.Symbol("b0"), sympy.Symbol("g0")]
    # Basic test whether circuit gives correct result
    obj_param_resolver = ising_obj.qaoa._get_param_resolver(ising_obj, 1, 0)
    wf_x = ising_obj.simulator.simulate(
        ising_obj.circuit, param_resolver=obj_param_resolver
    ).state_vector()
    assert np.allclose(
        wf_x,
        np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]),
        rtol=0,
        atol=1e-14,
    )
    obj_param_resolver = ising_obj.qaoa._get_param_resolver(ising_obj, 1, 0.5)
    wf_z = ising_obj.simulator.simulate(
        ising_obj.circuit, param_resolver=obj_param_resolver
    ).state_vector()
    assert np.allclose(
        wf_z,
        np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j]),
        rtol=0,
        atol=1e-14,
    )


# still needs to be improved
def test_set_p():
    ising_obj = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising_obj.set_circuit("qaoa", 1)
    assert ising_obj.p == 1
    ising_obj.qaoa.set_p(ising_obj, 2)
    assert ising_obj.p == 2


# test more examples
# test assertion error
def test_set_beta_values():
    ising_obj = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising_obj.set_circuit("qaoa", 2)
    ising_obj.qaoa._set_beta_values(ising_obj, [0.2, 0.3])
    # Assert whether self.circuit_param_values has been set correctly
    assert (ising_obj.circuit_param_values == np.array([0.2, 0.0, 0.3, 0.0])).all()


# test more examples
# test assertion error
def test_set_gamma_values():
    ising_obj = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising_obj.set_circuit("qaoa", 2)
    ising_obj.qaoa._set_gamma_values(ising_obj, [0.4, 0.5])
    # Assert whether self.circuit_param_values has been set correctly
    assert (ising_obj.circuit_param_values == np.array([0.0, 0.4, 0.0, 0.5])).all()


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


def test__get_param_resolver():
    self = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    # test for p = 1
    obj_param_resolver = self.qaoa._get_param_resolver(self, 0.1, 0.2)
    assert obj_param_resolver == cirq.ParamResolver({"b0": 0.1, "g0": 0.2})
    # test for p = 2
    self.p = 2
    obj_param_resolver = self.qaoa._get_param_resolver(self, [0.1, 0.3], [0.2, 0.4])
    assert obj_param_resolver == cirq.ParamResolver({"b0": 0.1, "g0": 0.2, "b1": 0.3, "g1": 0.4})


#############################################################
#                     Test errors                           #
#############################################################
def test_set_circuit_erros():
    ising_obj = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    with pytest.raises(AssertionError):
        ising_obj.set_circuit("test", 1)

    with pytest.raises(AssertionError):
        ising_obj.set_circuit("qaoa", [2, 3])


def test_set_p_errors():
    ising_obj = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising_obj.qaoa.set_p(ising_obj, 2)
    assert ising_obj.p == 2


def test_set_beta_values_errors():
    self = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    self.qaoa._set_beta_values(self, 1)
    assert self.p == 1
    assert (self.circuit_param_values == np.array((1, 0))).all()
    self.circuit_param_values = -1
    self.qaoa._set_beta_values(self, 1)
    assert (self.circuit_param_values == np.array((1, 0))).all()


def test_set_gamma_values_errors():
    self = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    self.qaoa._set_gamma_values(self, 1)
    assert self.p == 1
    assert (self.circuit_param_values == np.array((0, 1))).all()
    self.circuit_param_values = -1
    self.qaoa._set_gamma_values(self, 1)
    assert (self.circuit_param_values == np.array((0, 1))).all()


def test__get_param_resolver_errors():
    self = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    # test for p = 1
    with pytest.raises(AssertionError):
        obj_param_resolver = self.qaoa._get_param_resolver(self, 0.1, [0.2, 0.3])

    self.p = 2
    with pytest.raises(AssertionError):
        obj_param_resolver = self.qaoa._get_param_resolver(self, 0.1, 0.2)


# to check whether circuits are set correctly, simulate them (2 qubits) and compare unitrary
# wf1 = self.simulator.simulate(self.circuit, \
#        param_resolver = cirq.ParamResolver(joined_dict)).state_vector()

# add parametrisation of tests.
"""
@pytest.mark.parametrize(
    'qubittype, n, j_v, j_h, h, test_gate, E_exp', 
    [
        #############################################################
        #                   2 qubit tests                           #
        #############################################################
        #Normalised energy; hence expect -j_v/2 = -0.1:
        ('GridQubit', [1, 2], np.ones((0,2))/2, np.ones((1,1))/5, \
            np.zeros((1,2))/10, cirq.X, -0.1),
        #Normalised energy; hence expect -j_v/2 = -0.1:
        ('GridQubit', [1, 2], np.ones((0,2))/2, np.ones((1,1))/5, \
            np.zeros((1,2))/10, cirq.X**2, -0.1),
        #Normalised energy; hence expect -j_h/2 = -0.25
        ('GridQubit', [2, 1], np.ones((1,1))/2, np.ones((2,0))/5, \
            np.zeros((2,1))/10, cirq.X, -0.25),
        #Normalised energy; hence expect -j_h/2 = -0.25
        ('GridQubit', [2, 1], np.ones((1,1))/2, np.ones((2,0))/5, \
            np.zeros((2,1))/10, cirq.X**2, -0.25),
        #Normalised energy; hence expect -j_v/2 - 2h/2= -0.2:
        ('GridQubit', [1, 2], np.ones((0,2))/2, np.ones((1,1))/5, \
            np.ones((1,2))/10, cirq.X, -0.2),
        #Normalised energy; hence expect -j_v/2 + 2h/2 = 0
        ('GridQubit', [1, 2], np.ones((0,2))/2, np.ones((1,1))/5, \
            np.ones((1,2))/10, cirq.X**2, 0),
        #Normalised energy; hence expect (-j_h-2h)/2 = -0.35
        ('GridQubit', [2, 1], np.ones((1,1))/2, np.ones((2,0))/5, \
            np.ones((2,1))/10, cirq.X, -0.35),
        #Normalised energy; hence expect (-j_h+2h)/2 = -0.15
        ('GridQubit', [2, 1], np.ones((1,1))/2, np.ones((2,0))/5, \
            np.ones((2,1))/10, cirq.X**2, -0.15),
        #############################################################
        #                   4 qubit tests                           #
        #############################################################
        #Normalised energy; hence expect (-2j_h-2j_v-4h)/4 = -0.45
        ('GridQubit', [2, 2], np.ones((1,2))/2, np.ones((2,1))/5, \
            np.ones((2,2))/10, cirq.X, -0.45),
        #Normalised energy; hence expect (-2j_h-2j_v-4h)/4 = -0.45
        ('GridQubit', [2, 2], np.ones((1,2))/2, np.ones((2,1))/5, \
            np.ones((2,2))/10, cirq.X**2, -0.25),
    ]
)
def test_energy(qubittype, n, j_v, j_h, h, test_gate, E_exp):
    #set numerical tolerance
    atol = 1E-14
    tester = IsingTester(atol)
    tester.simple_energy_test(qubittype, n, j_v, j_h, h, test_gate, E_exp)

@pytest.mark.parametrize(
    'qubittype, n, j_v, j_h, h, test_gate, vm_exp, apply_to', 
    [
        #############################################################
        #                   1 qubit tests                           #
        #############################################################
        ('GridQubit', [1, 1], np.zeros((0,1))/2, np.zeros((1,0))/5, np.zeros((1,1))/10, \
            cirq.Z, {(0, 0): -1.0}, []),
        #############################################################
        #                   2 qubit tests                           #
        #############################################################
        ('GridQubit', [1, 2], np.zeros((0,2))/2, np.zeros((1,1))/5, np.zeros((1,2))/10, \
            cirq.Z, {(0, 0): -1.0, (0, 1): -1.0}, []),
        ('GridQubit', [2, 1], np.zeros((1,1))/2, np.zeros((2,0))/5, np.zeros((2,1))/10, \
            cirq.Z, {(0, 0): -1.0, (1, 0): -1.0}, []),            
        #############################################################
        #                   4 qubit tests                           #
        #############################################################
        # Okay this Z is just a phase gate:
        ('GridQubit', [2, 2], np.zeros((1,2))/2, np.zeros((2,1))/5, np.zeros((2,2))/10, \
            cirq.Z, {(0, 0): -1.0, (0, 1): -1.0, (1, 0): -1.0, (1, 1): -1.0}, []),
        ('GridQubit', [2, 2], np.zeros((1,2))/2, np.zeros((2,1))/5, np.zeros((2,2))/10, \
            cirq.Z**2, {(0, 0): -1.0, (0, 1): -1.0, (1, 0): -1.0, (1, 1): -1.0}, []),
        # X is spin flip |0000> -> |1111>:
        ('GridQubit', [2, 2], np.zeros((1,2))/2, np.zeros((2,1))/5, np.zeros((2,2))/10, \
            cirq.X, {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): 1.0}, []),
        # H : |0000> -> 1/\sqrt(2)**(n/2) \sum_i=0^2**1-1 |i>
        ('GridQubit', [2, 2], np.zeros((1,2))/2, np.zeros((2,1))/5, np.zeros((2,2))/10, \
            cirq.H, {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0}, []),
        # Test whether numbering is correct
        ('GridQubit', [2, 2], np.zeros((1,2))/2, np.zeros((2,1))/5, np.zeros((2,2))/10, \
            cirq.X, {(0, 0): 1.0, (0, 1): -1.0, (1, 0): -1.0, (1, 1): -1.0}, np.array([[1,0], [0,0]])),
        ('GridQubit', [2, 2], np.zeros((1,2))/2, np.zeros((2,1))/5, np.zeros((2,2))/10, \
            cirq.X, {(0, 0): -1.0, (0, 1): 1.0, (1, 0): -1.0, (1, 1): -1.0}, np.array([[0,1], [0,0]])),
        ('GridQubit', [2, 2], np.zeros((1,2))/2, np.zeros((2,1))/5, np.zeros((2,2))/10, \
            cirq.X, {(0, 0): -1.0, (0, 1): -1.0, (1, 0): 1.0, (1, 1): -1.0}, np.array([[0,0], [1,0]])),
        ('GridQubit', [2, 2], np.zeros((1,2))/2, np.zeros((2,1))/5, np.zeros((2,2))/10, \
            cirq.X, {(0, 0): -1.0, (0, 1): -1.0, (1, 0): -1.0, (1, 1): 1.0}, np.array([[0,0], [0,1]])),
    ]
)
def test_get_spin_vm(qubittype, n, j_v, j_h, h, test_gate, vm_exp, apply_to):
    #set numerical tolerance
    atol = 1E-14
    tester = IsingTester(atol)
    tester.simple_spin_value_map_test(qubittype, n, j_v, j_h, h, test_gate, vm_exp, apply_to)

#Missing: test print_spin properly
def test_print_spin_dummy():
    ising_obj = Ising('GridQubit', [2, 2], np.zeros((1,2))/2, np.zeros((2,1))/5, np.zeros((2,2))/10)
    
    # Dummy to generate 'empty circuit'
    for i in range(ising_obj.n[0]):
      for j in range(ising_obj.n[1]):
        ising_obj.circuit.append(cirq.Z(ising_obj.qubits[i][j])**2)

    
    wf = ising_obj.simulator.simulate(ising_obj.circuit).state_vector()
    ising_obj.print_spin(wf)
    """
