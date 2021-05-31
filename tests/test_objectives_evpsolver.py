"""
    Test EVPSolver()

    -test some simple examples 
    -test self consistency -> compare output of different analytic solvers

"""
# external imports
import pytest
import numpy as np
import scipy
from cirq.testing import lin_alg_utils

# internal imports
from fauvqe import EVPSolver, Ising

@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, val_exp, vec_exp",
    [
        (
            "GridQubit",
            [2, 1],
            np.zeros((1, 1)),
            np.zeros((2, 0)),
            np.array([0.5, 1]).reshape((2,1)), 
            [-0.75, -0.25],
            np.transpose([
                [-0.5, -0.5, -0.5, -0.5],
                [ 0.5,  0.5, -0.5, -0.5],
            ]),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((1, 2)),
            np.ones((2, 1)),
            np.zeros((2, 2)),
            [-1, -1],
            np.transpose([
                [1.+0.j , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0       ],
                [0      , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.+0.j  ],
            ]),
        ),
    ]
)

def test_evaluate(qubittype, n, j_v, j_h, h, val_exp, vec_exp):
    # Create Ising object
    ising_obj = Ising(qubittype, n, j_v, j_h, h)

    #Calculate analytic results by different methods
    np_sol = EVPSolver(ising_obj)
    np_sol.evaluate(solver = 'numpy')

    scipy_sol = EVPSolver(ising_obj)
    scipy_sol.evaluate(solver = 'scipy')

    sparse_scipy_sol = EVPSolver(ising_obj)
    sparse_scipy_sol.evaluate()

    # Test whether found eigenvalues are all close up to tolerance
    np.testing.assert_allclose(scipy_sol.val    , sparse_scipy_sol.val, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(np_sol.val[0:2]  , sparse_scipy_sol.val, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(val_exp          , sparse_scipy_sol.val, rtol=1e-15, atol=1e-15)

    # Test whether found eigenvectors are all close up to tolerance and global phase
    # Note that different eigen vectors can have a different global phase; hence we assert them one by one
    # Here we only check ground state and first excited state
    # Further issue: abitrary for degenerate
    for i in range(2):
        if np.abs(sparse_scipy_sol.val[0] - sparse_scipy_sol.val[1]) > 1e-14:
            lin_alg_utils.assert_allclose_up_to_global_phase(scipy_sol.vec[:,i] , sparse_scipy_sol.vec[:,i], rtol=1e-15, atol=1e-15)
        
        lin_alg_utils.assert_allclose_up_to_global_phase(np_sol.vec[:,i]    , scipy_sol.vec[:,i], rtol=1e-15, atol=1e-15)
        lin_alg_utils.assert_allclose_up_to_global_phase(vec_exp[:,i]       , scipy_sol.vec[:,i], rtol=1e-15, atol=1e-15)

def test_evaluate_erros():
    # Create Ising object as an example of AbstractModel
    ising_obj = Ising("GridQubit",
            [2, 2],
            np.ones((1, 2)),
            np.ones((2, 1)),
            np.zeros((2, 2)))

    evp_solver = EVPSolver(ising_obj)
    with pytest.raises(AssertionError):
        evp_solver.evaluate(solver="numpy.sparse")

def test_repr():
    # Create Ising object as an example of AbstractModel
    ising_obj = Ising("GridQubit",
            [2, 2],
            np.ones((1, 2)),
            np.ones((2, 1)),
            np.zeros((2, 2)))

    evp_solver = EVPSolver(ising_obj)
    assert repr(evp_solver) ==  "<EVPSolver>" 