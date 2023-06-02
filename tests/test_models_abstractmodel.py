"""
Test parent class AbstractModel;
    -test whether correct error messages occur
    -test whether initialisation set obj.qubits correctly 
        based on some examples
    -try to check test coverage?
"""
# external import
import cirq
import qsimcirq
import pytest
import numpy as np
import sympy
from timeit import default_timer

# internal import
from fauvqe import AbstractModel, Converter


class MockAbstractModel(AbstractModel):
    def copy(self):
        return MockAbstractModel()

    def energy(self):
        return np.array([])

    def from_json_dict(self, dct):
        return MockInitialiser()

    def to_json_dict(self):
        return {}

    def _set_hamiltonian(self, reset: bool = True):
        self._hamiltonian = cirq.PauliSum()

def test__eq__():
    model1 = MockAbstractModel("GridQubit", [2, 2])
    model2 = MockAbstractModel("GridQubit", [2, 2])
    assert (model1 == model2)

    model1.t = 1
    assert (model1 != model2)

    non_model = dict()
    assert (model1 != non_model)

# test_AbstractModel_set_qubits
@pytest.mark.parametrize(
    "qubittype, n, exp_quibits",
    [
        ("NamedQubit", "a", [cirq.NamedQubit("a")]),
        ("NamedQubit", ["a", "b"], [cirq.NamedQubit("a"), cirq.NamedQubit("b")]),
        ("LineQubit", 1, [cirq.LineQubit(0)]),
        # This should work but doesn't:
        # ('LineQubit', np.array(1), [cirq.LineQubit(0)]),
        ("LineQubit", 2, [cirq.LineQubit(0), cirq.LineQubit(1)]),
        ("GridQubit", np.array([1, 1]), [[cirq.GridQubit(0, 0)]]),
        ("GridQubit", [1, 1], [[cirq.GridQubit(0, 0)]]),
        ("GridQubit", [2, 1], [[cirq.GridQubit(0, 0)], [cirq.GridQubit(1, 0)]]),
        ("GridQubit", [1, 2], [[cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]]),
    ],
)
def test_AbstractModel(qubittype, n, exp_quibits):
    AbstractModel_obj = MockAbstractModel(qubittype, n)
    assert AbstractModel_obj.qubits == exp_quibits
    assert AbstractModel_obj.qubittype == qubittype
    if isinstance(n, np.ndarray):
        assert (AbstractModel_obj.n == n).all()
    else:
        assert AbstractModel_obj.n == n


# test whether circuit and simulator was created
def test_AbstractModel_exist():
    AbstractModel_obj = MockAbstractModel("LineQubit", 1)
    assert hasattr(AbstractModel_obj, "simulator")
    assert hasattr(AbstractModel_obj, "qubits")


# test whether circuit and simulator was created
def test_set_simulator():
    AbstractModel_obj = MockAbstractModel("LineQubit", 1)
    # Check if default parameter is given
    assert type(AbstractModel_obj.simulator) == qsimcirq.qsim_simulator.QSimSimulator
    assert AbstractModel_obj.simulator_options == {"t": 8, "f": 4}
    # Check whether adding parameters works for qsim
    AbstractModel_obj.set_simulator(simulator_options={"f": 2})
    assert AbstractModel_obj.simulator_options == {"t": 8, "f": 2}

    # Check whether cirq simulator can be set
    # and whether simulator_options are correct default
    AbstractModel_obj.set_simulator(simulator_name="cirq")
    assert type(AbstractModel_obj.simulator) == cirq.sim.sparse_simulator.Simulator
    assert AbstractModel_obj.simulator_options == {"dtype": np.complex64}

    # Test whether an Assertion error is raised otherwise
    with pytest.raises(AssertionError):
        AbstractModel_obj.set_simulator(simulator_name="simulator")


@pytest.mark.parametrize(
    "qubittype, n, coefficients, gates, qubits, val_exp, vec_exp",
    [
        (
            "GridQubit",
            [2, 1],
            [-0.5, -1],
            [cirq.X,  cirq.X],
            [[0, 0], [1, 0]],
            [-0.75, -0.25],
            np.transpose([
                [-0.5, -0.5, -0.5, -0.5],
                [ 0.5,  0.5, -0.5, -0.5],
            ]),
        ),
        (
            "GridQubit",
            [2, 1],
            [0.5, -1],
            [cirq.X,  cirq.Z],
            [[0, 0], [1, 0]],
            [-0.75, -0.25],
            np.transpose([
                [ np.sqrt(2)/2,  0, -np.sqrt(2)/2, 0],
                [ np.sqrt(2)/2,  0, np.sqrt(2)/2, 0],
            ]),
        ),
#        (
#            "GridQubit",
#            [1, 3],
#            [2, -0.5, 1],
#            [cirq.X,  cirq.Y ,cirq.Z],
#            [[0, 0], [0, 1], [0, 2]],
#            [-7/6, -5/6],
#            np.transpose([
                #[0, 0.5, 0, 0.5j, 0, -0.5, 0, -0.5j],
#                [0, -0.5j, 0, -0.5, 0, 0.5j, 0, 0.5],
#                [0, 0.5, 0, 0.5, 0, 0.5, 0, 1],
#            ]),
#        ),
        (
            "LineQubit",
            2,
            [-0.5, -1],
            [cirq.X,  cirq.X],
            [0, 1],
            [-0.75, -0.25],
            np.transpose([
                [-0.5, -0.5, -0.5, -0.5],
                [ 0.5,  0.5, -0.5, -0.5],
            ]),
        ),
        (
            "LineQubit",
            2,
            [0.5, -1],
            [cirq.X,  cirq.Z],
            [0, 1],
            [-0.75, -0.25],
            np.transpose([
                [ np.sqrt(2)/2,  0, -np.sqrt(2)/2, 0],
                [ np.sqrt(2)/2,  0, np.sqrt(2)/2, 0],
            ]),
        ),
        (
            "NamedQubit",
            ["a","b"],
            [-0.5, -1],
            [cirq.X,  cirq.X],
            [0, 1],
            [-0.75, -0.25],
            np.transpose([
                [-0.5, -0.5, -0.5, -0.5],
                [ 0.5,  0.5, -0.5, -0.5],
            ]),
        ),
        (
            "NamedQubit",
            ["a","b"],
            [0.5, -1],
            [cirq.X,  cirq.Z],
            [0, 1],
            [-0.75, -0.25],
            np.transpose([
                [ np.sqrt(2)/2,  0, -np.sqrt(2)/2, 0],
                [ np.sqrt(2)/2,  0, np.sqrt(2)/2, 0],
            ]),
        ),
    ]
)
def test_diagonalise(qubittype, n, coefficients, gates, qubits, val_exp, vec_exp):
    # Create AbstractModel object
    np_sol = MockAbstractModel(qubittype, n)
    scipy_sol = MockAbstractModel(qubittype, n)
    sparse_scipy_sol = MockAbstractModel(qubittype, n)

    #Create the hamiltonians
    for i in range(np.size(gates)):
        gate = gates[i]
        if qubittype == "GridQubit":
            np_sol._hamiltonian += coefficients[i]*gate(np_sol.qubits[qubits[i][0]][qubits[i][1]])
            scipy_sol._hamiltonian += coefficients[i]*gate(np_sol.qubits[qubits[i][0]][qubits[i][1]])
            sparse_scipy_sol._hamiltonian += coefficients[i]*gate(np_sol.qubits[qubits[i][0]][qubits[i][1]])
        else:
            np_sol._hamiltonian += coefficients[i]*gate(np_sol.qubits[qubits[i]])
            scipy_sol._hamiltonian += coefficients[i]*gate(np_sol.qubits[qubits[i]])
            sparse_scipy_sol._hamiltonian += coefficients[i]*gate(np_sol.qubits[qubits[i]])

    #Calculate analytic results by different methods
    np_sol.diagonalise(solver = 'numpy')
    scipy_sol.diagonalise(solver = 'scipy')
    sparse_scipy_sol.diagonalise()

    # Test whether found eigenvalues are all close up to tolerance
    np.testing.assert_allclose(scipy_sol.eig_val    , sparse_scipy_sol.eig_val, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(np_sol.eig_val[0:2]  , sparse_scipy_sol.eig_val, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(val_exp          , sparse_scipy_sol.eig_val, rtol=1e-15, atol=1e-15)

    # Test whether found eigenvectors are all close up to tolerance and global phase
    # Note that different eigen vectors can have a different global phase; hence we assert them one by one
    # Here we only check ground state and first excited state
    # Further issue: abitrary for degenerate
    for i in range(2):
        if np.abs(sparse_scipy_sol.eig_val[0] - sparse_scipy_sol.eig_val[1]) > 1e-14:
            #assert(sparse_scipy_sol.val[0] == sparse_scipy_sol.val[1] )
            cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(scipy_sol.eig_vec[:,i] , sparse_scipy_sol.eig_vec[:,i], rtol=1e-15, atol=1e-15)
        
        cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(np_sol.eig_vec[:,i]    , scipy_sol.eig_vec[:,i], rtol=1e-15, atol=1e-15)
        cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(vec_exp[:,i]       , scipy_sol.eig_vec[:,i], rtol=1e-15, atol=1e-15)

@pytest.mark.parametrize(
    "paulisum",
    [
        (
            cirq.PauliSum.from_pauli_strings([-np.pi*cirq.Z(cirq.LineQubit(i))*cirq.Z(cirq.LineQubit(i+1)) for i in range(8)]) +
            cirq.PauliSum.from_pauli_strings([-i*cirq.Z(cirq.LineQubit(i)) for i in range(8)]) + 
            cirq.PauliSum.from_pauli_strings([-cirq.X(cirq.LineQubit(i))/(i+1) for i in range(8)])
        ),
    ]
)
def test_diagonalise_dense_sparse_speedup(paulisum):
    np_sol = MockAbstractModel("LineQubit", 2)
    sparse_scipy_sol = MockAbstractModel("LineQubit", 2)

    convert_obj = Converter()
    scipy_sparse_matrix = convert_obj.cirq_paulisum2scipy_crsmatrix(paulisum)

    t0 = default_timer()
    np_sol.diagonalise(solver = 'numpy', matrix = paulisum.matrix())
    t1 = default_timer()
    sparse_scipy_sol.diagonalise(matrix = scipy_sparse_matrix) 
    t2 = default_timer()
    print("Numpy dense: {}\tScipy Sparse: {}\tSpeed-up: {}".format(t1-t0, t2-t1, (t1-t0)/(t2-t1)))

    #Test speed up
    assert((t1-t0)/(t2-t1) > 25)

    # Test whether found eigenvalues are all close up to tolerance
    np.testing.assert_allclose(np_sol.eig_val[0:2]  , sparse_scipy_sol.eig_val, rtol=1e-13, atol=1e-14)

    # Test whether found eigenvectors are all close up to tolerance and global phase
    for i in range(2):
        if np.abs(sparse_scipy_sol.eig_val[0] - sparse_scipy_sol.eig_val[1]) > 1e-14:
            cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(np_sol.eig_vec[:,i]    , sparse_scipy_sol.eig_vec[:,i], rtol=1e-14, atol=1e-13)

@pytest.mark.parametrize(
    "n, circuit, options, solution_circuit",
    [
        (   
            [1,1],
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0))),
            dict(),
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 0))),
        ),
        (   
            [1,2],
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), 
                        (cirq.ZZ**(sympy.Symbol('a'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                ),
            dict(),
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), 
                        (cirq.ZZ**(sympy.Symbol('a'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(sympy.Symbol('a'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.H.on(cirq.GridQubit(0, 0)), 
            ),
        ),
        (   
            [2,1],
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), 
                        (cirq.ZZ**(sympy.Symbol('a'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                ),
            {'start': 0, 'end': 1},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), 
                        (cirq.ZZ**(sympy.Symbol('a'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        cirq.H.on(cirq.GridQubit(0, 0)), 
                        (cirq.ZZ**(sympy.Symbol('a'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
        ),
        (   
            [2,2],
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.X.on(cirq.GridQubit(1, 0)), cirq.Y.on(cirq.GridQubit(0, 1)), cirq.Z.on(cirq.GridQubit(1, 1)),
                        (cirq.ZZ**(sympy.Symbol('a'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.YY**(sympy.Symbol('b'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        (cirq.XX**(sympy.Symbol('c'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.CNOT.on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
                ),
            {'start': 2, 'end': 1},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.X.on(cirq.GridQubit(1, 0)), cirq.Y.on(cirq.GridQubit(0, 1)), cirq.Z.on(cirq.GridQubit(1, 1)),
                        (cirq.ZZ**(sympy.Symbol('a'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.YY**(sympy.Symbol('b'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
                        (cirq.XX**(sympy.Symbol('c'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.CNOT.on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
                        (cirq.XX**(sympy.Symbol('c'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.CNOT.on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
                        (cirq.ZZ**(sympy.Symbol('a'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.YY**(sympy.Symbol('b'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
            ),
        ),
    ]
)
def test_symmetrise_circuit(n, circuit, options, solution_circuit):
    model = MockAbstractModel("GridQubit", n)
    model.circuit=circuit

    print(model.circuit)
    model.symmetrise_circuit(options)
    print(model.circuit)

    assert(model.circuit == solution_circuit)

@pytest.mark.parametrize(
    "model, error",
    [
        (
            MockAbstractModel("LineQubit", 3),
            NotImplementedError,
        ),
        (
            MockAbstractModel("GridQubit", [1,3]),
            AssertionError,
        ),
    ]
)
def test_glue_circuit_error(model, error):
    with pytest.raises(error):
        model.glue_circuit()

@pytest.mark.parametrize(
    "n, axis, repetitions, init_circuit, init_circuit_param, init_circuit_param_values, final_circuit, final_circuit_param, final_circuit_param_values",
    [
        (
            np.array([1,3]),
            1,
            2,
            cirq.Circuit((cirq.ZZ**sympy.Symbol('a')).on(cirq.GridQubit(0,0),cirq.GridQubit(0,1)),
                         (cirq.ZZ**sympy.Symbol('b')).on(cirq.GridQubit(0,1),cirq.GridQubit(0,2)),
                         (cirq.ZZ**sympy.Symbol('c')).on(cirq.GridQubit(0,2),cirq.GridQubit(0,0))),
            [sympy.Symbol('a'),sympy.Symbol('b'),sympy.Symbol('c')],
            np.array([1, 1/2, 1/3]),
            cirq.Circuit((cirq.ZZ**sympy.Symbol('a_g0')).on(cirq.GridQubit(0,0),cirq.GridQubit(0,1)),
                         (cirq.ZZ**sympy.Symbol('a_g1')).on(cirq.GridQubit(0,3),cirq.GridQubit(0,4)),
                         (cirq.ZZ**sympy.Symbol('b_g0')).on(cirq.GridQubit(0,1),cirq.GridQubit(0,2)),
                         (cirq.ZZ**sympy.Symbol('b_g1')).on(cirq.GridQubit(0,4),cirq.GridQubit(0,5)),
                         (cirq.ZZ**sympy.Symbol('c_g0')).on(cirq.GridQubit(0,2),cirq.GridQubit(0,3)),
                         (cirq.ZZ**sympy.Symbol('c_g1')).on(cirq.GridQubit(0,5),cirq.GridQubit(0,0))),
            [sympy.Symbol('a_g0'),sympy.Symbol('b_g0'),
             sympy.Symbol('c_g0'),sympy.Symbol('a_g1'),
             sympy.Symbol('b_g1'),sympy.Symbol('c_g1')],
            np.array([1,1,1/2, 1/2, 1/3,1/3]),
        ),
        (
            np.array([1,3]),
            1,
            2,
            cirq.Circuit((cirq.ZZ**sympy.Symbol('a')).on(cirq.GridQubit(0,0),cirq.GridQubit(0,1)),
                         (cirq.ZZ**0.4).on(cirq.GridQubit(0,1),cirq.GridQubit(0,2)),
                         (cirq.ZZ**sympy.Symbol('c')).on(cirq.GridQubit(0,2),cirq.GridQubit(0,0))),
            [sympy.Symbol('a'),sympy.Symbol('c')],
            np.array([1, 1/3]),
            cirq.Circuit((cirq.ZZ**sympy.Symbol('a_g0')).on(cirq.GridQubit(0,0),cirq.GridQubit(0,1)),
                         (cirq.ZZ**sympy.Symbol('a_g1')).on(cirq.GridQubit(0,3),cirq.GridQubit(0,4)),
                         (cirq.ZZ**0.4).on(cirq.GridQubit(0,1),cirq.GridQubit(0,2)),
                         (cirq.ZZ**0.4).on(cirq.GridQubit(0,4),cirq.GridQubit(0,5)),
                         (cirq.ZZ**sympy.Symbol('c_g0')).on(cirq.GridQubit(0,2),cirq.GridQubit(0,3)),
                         (cirq.ZZ**sympy.Symbol('c_g1')).on(cirq.GridQubit(0,5),cirq.GridQubit(0,0))),
            [sympy.Symbol('a_g0'),sympy.Symbol('c_g0'),
             sympy.Symbol('a_g1'),sympy.Symbol('c_g1')],
            np.array([1,1, 1/3,1/3]),
        ),
        (
            np.array([1,3]),
            1,
            2,
            cirq.Circuit( cirq.H.on(cirq.GridQubit(0,0)),cirq.H.on(cirq.GridQubit(0,1)),cirq.H.on(cirq.GridQubit(0,2)),
                            (cirq.ZZ**sympy.Symbol('a')).on(cirq.GridQubit(0,0),cirq.GridQubit(0,1)),
                         (cirq.ZZ**sympy.Symbol('b')).on(cirq.GridQubit(0,1),cirq.GridQubit(0,2)),
                         (cirq.ZZ**sympy.Symbol('c')).on(cirq.GridQubit(0,2),cirq.GridQubit(0,0))),
            [sympy.Symbol('a'),sympy.Symbol('b'),sympy.Symbol('c')],
            np.array([1, 1/2, 1/3]),
            cirq.Circuit(   cirq.H.on(cirq.GridQubit(0,0)),cirq.H.on(cirq.GridQubit(0,1)),cirq.H.on(cirq.GridQubit(0,2)),
                            cirq.H.on(cirq.GridQubit(0,3)),cirq.H.on(cirq.GridQubit(0,4)),cirq.H.on(cirq.GridQubit(0,5)),
                            (cirq.ZZ**sympy.Symbol('a_g0')).on(cirq.GridQubit(0,0),cirq.GridQubit(0,1)),
                         (cirq.ZZ**sympy.Symbol('a_g1')).on(cirq.GridQubit(0,3),cirq.GridQubit(0,4)),
                         (cirq.ZZ**sympy.Symbol('b_g0')).on(cirq.GridQubit(0,1),cirq.GridQubit(0,2)),
                         (cirq.ZZ**sympy.Symbol('b_g1')).on(cirq.GridQubit(0,4),cirq.GridQubit(0,5)),
                         (cirq.ZZ**sympy.Symbol('c_g0')).on(cirq.GridQubit(0,2),cirq.GridQubit(0,3)),
                         (cirq.ZZ**sympy.Symbol('c_g1')).on(cirq.GridQubit(0,5),cirq.GridQubit(0,0))),
            [sympy.Symbol('a_g0'),sympy.Symbol('b_g0'),
             sympy.Symbol('c_g0'),sympy.Symbol('a_g1'),
             sympy.Symbol('b_g1'),sympy.Symbol('c_g1')],
            np.array([1,1,1/2, 1/2, 1/3,1/3]),
        ),
        (
            np.array([3,1]),
            0,
            3,
            cirq.Circuit((cirq.XX**sympy.Symbol('a')).on(cirq.GridQubit(0,0),cirq.GridQubit(1,0)),
                         (cirq.YY**sympy.Symbol('b')).on(cirq.GridQubit(1,0),cirq.GridQubit(2,0)),
                         (cirq.ZZ**sympy.Symbol('c')).on(cirq.GridQubit(2,0),cirq.GridQubit(0,0))),
            [sympy.Symbol('a'),sympy.Symbol('b'),sympy.Symbol('c')],
            np.array([1, 1/2, 1/3]),
            cirq.Circuit((cirq.XX**sympy.Symbol('a_g0')).on(cirq.GridQubit(0,0),cirq.GridQubit(1,0)),
                         (cirq.XX**sympy.Symbol('a_g1')).on(cirq.GridQubit(3,0),cirq.GridQubit(4,0)),
                         (cirq.XX**sympy.Symbol('a_g2')).on(cirq.GridQubit(6,0),cirq.GridQubit(7,0)),
                         (cirq.YY**sympy.Symbol('b_g0')).on(cirq.GridQubit(1,0),cirq.GridQubit(2,0)),
                         (cirq.YY**sympy.Symbol('b_g1')).on(cirq.GridQubit(4,0),cirq.GridQubit(5,0)),
                         (cirq.YY**sympy.Symbol('b_g2')).on(cirq.GridQubit(7,0),cirq.GridQubit(8,0)),
                         (cirq.ZZ**sympy.Symbol('c_g0')).on(cirq.GridQubit(2,0),cirq.GridQubit(3,0)),
                         (cirq.ZZ**sympy.Symbol('c_g1')).on(cirq.GridQubit(5,0),cirq.GridQubit(6,0)),
                         (cirq.ZZ**sympy.Symbol('c_g2')).on(cirq.GridQubit(8,0),cirq.GridQubit(0,0))),
            [sympy.Symbol('a_g0'),sympy.Symbol('b_g0'),sympy.Symbol('c_g0'),
             sympy.Symbol('a_g1'),sympy.Symbol('b_g1'),sympy.Symbol('c_g1'),
             sympy.Symbol('a_g2'),sympy.Symbol('b_g2'),sympy.Symbol('c_g2'),],
            np.array([1,1,1,1/2,1/2, 1/2, 1/3,1/3,1/3]),
        ),
        (
            np.array([1,3]),
            1,
            2,
            cirq.Circuit((cirq.ZZ**sympy.Symbol('a')).on(cirq.GridQubit(0,0),cirq.GridQubit(0,1)),
                         (cirq.ZZ**sympy.Symbol('b')).on(cirq.GridQubit(0,1),cirq.GridQubit(0,2)),
                         (cirq.ZZ**sympy.Symbol('c')).on(cirq.GridQubit(0,2),cirq.GridQubit(0,0)),
                         cirq.H.on(cirq.GridQubit(0,0)),cirq.H.on(cirq.GridQubit(0,1)),cirq.H.on(cirq.GridQubit(0,2))),
            [sympy.Symbol('a'),sympy.Symbol('b'),sympy.Symbol('c')],
            np.array([1, 1/2, 1/3]),
            cirq.Circuit( (cirq.ZZ**sympy.Symbol('a_g0')).on(cirq.GridQubit(0,0),cirq.GridQubit(0,1)),
                         (cirq.ZZ**sympy.Symbol('a_g1')).on(cirq.GridQubit(0,3),cirq.GridQubit(0,4)),
                         (cirq.ZZ**sympy.Symbol('b_g0')).on(cirq.GridQubit(0,1),cirq.GridQubit(0,2)),
                         (cirq.ZZ**sympy.Symbol('b_g1')).on(cirq.GridQubit(0,4),cirq.GridQubit(0,5)),
                         (cirq.ZZ**sympy.Symbol('c_g0')).on(cirq.GridQubit(0,2),cirq.GridQubit(0,3)),
                         (cirq.ZZ**sympy.Symbol('c_g1')).on(cirq.GridQubit(0,5),cirq.GridQubit(0,0)),
                         cirq.H.on(cirq.GridQubit(0,0)),cirq.H.on(cirq.GridQubit(0,1)),cirq.H.on(cirq.GridQubit(0,2)),
                        cirq.H.on(cirq.GridQubit(0,3)),cirq.H.on(cirq.GridQubit(0,4)),cirq.H.on(cirq.GridQubit(0,5)),),
            [sympy.Symbol('a_g0'),sympy.Symbol('b_g0'),
             sympy.Symbol('c_g0'),sympy.Symbol('a_g1'),
             sympy.Symbol('b_g1'),sympy.Symbol('c_g1')],
            np.array([1,1,1/2, 1/2, 1/3,1/3]),
        ),
    ]
)
def test_glue_circuit(n, axis, repetitions, 
                      init_circuit, init_circuit_param, init_circuit_param_values,
                      final_circuit, final_circuit_param, final_circuit_param_values):
    model = MockAbstractModel("GridQubit", n)
    model.circuit=init_circuit
    model.circuit_param = init_circuit_param
    model.circuit_param_values = init_circuit_param_values

    print(n)
    print(model.circuit)
    model.glue_circuit(axis, repetitions)

    if axis:
        n2 = n*[1,repetitions]
    else:
        n2 = n*[repetitions,1]

    
    print(final_circuit)

    assert (model.n == n2).all()
    assert model.circuit == final_circuit
    assert model.circuit_param == final_circuit_param
    np.allclose(model.circuit_param_values, final_circuit_param_values, rtol=0, atol=1e-15)

def test_flattened_qubits():
    model = MockAbstractModel("LineQubit", 2)
    assert model.qubits==model.flattened_qubits