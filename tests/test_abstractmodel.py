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

# internal import
from fauvqe import AbstractModel


class MockAbstractModel(AbstractModel):
    def to_json_dict(self):
        return {}

    def from_json_dict(self, dct):
        return MockInitialiser()

    def energy(self):
        return np.array([])

    def _set_hamiltonian(self, reset: bool = True):
        self.hamiltonian = cirq.PauliSum()


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
    assert AbstractModel_obj.simulator_options == {}

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
            np_sol.hamiltonian += coefficients[i]*gate(np_sol.qubits[qubits[i][0]][qubits[i][1]])
            scipy_sol.hamiltonian += coefficients[i]*gate(np_sol.qubits[qubits[i][0]][qubits[i][1]])
            sparse_scipy_sol.hamiltonian += coefficients[i]*gate(np_sol.qubits[qubits[i][0]][qubits[i][1]])
        else:
            np_sol.hamiltonian += coefficients[i]*gate(np_sol.qubits[qubits[i]])
            scipy_sol.hamiltonian += coefficients[i]*gate(np_sol.qubits[qubits[i]])
            sparse_scipy_sol.hamiltonian += coefficients[i]*gate(np_sol.qubits[qubits[i]])

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
