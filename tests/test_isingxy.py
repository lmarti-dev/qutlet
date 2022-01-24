# external imports
import pytest
import numpy as np
import cirq
from scipy.linalg import expm
import sympy

# internal imports
from fauvqe import IsingXY

def test__eq__():
    n = [1,3]; boundaries = [1, 0]
    ising = IsingXY("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])),"Z")
    ising.set_circuit("qaoa")
    
    ising2 = IsingXY("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])),"Z")
    ising2.set_circuit("qaoa")

    #print("ising == ising2: \t {}".format(ising == ising2))
    assert (ising == ising2)

    ising.set_Ut()
    assert ising != ising2 


@pytest.mark.parametrize(
    "qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h, basis",
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
            np.zeros((1, 2)) / 10,
            "Z",
        ),
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.zeros((1, 2)) / 10,
            "Z",
        ),
        (
            "GridQubit",
            [2, 1],
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            "Z",
        ),
        (
            "GridQubit",
            [2, 1],
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            "Z",
        ),
        (
            "GridQubit",
            [1, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((1, 2)) / 5,
            np.zeros((1, 2)) / 2,
            np.zeros((1, 2)) / 5,
            np.ones((1, 2)) / 3,
            "X",
        ),
        (
            "GridQubit",
            [1, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((1, 2)) / 5,
            np.zeros((1, 2)) / 2,
            np.zeros((1, 2)) / 5,
            np.ones((1, 2)) / 3,
            "X",
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
            np.ones((2, 2)) / 3,
            "X",
        ),
        (
            "GridQubit",
            [2, 2],
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.ones((2, 2)) / 3,
            "X",
        ),
        (
            "GridQubit",
            [2, 2],
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.ones((2, 2)) / 3,
            "X",
        ),
    ],
)
def test_copy(qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h, basis):
    ising = IsingXY(qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h, basis)
    ising.set_circuit("qaoa")
    ising2 = ising.copy()

    #Test whether the objects are the same
    assert( ising == ising2 )
    
    #But there ID is different
    assert( ising is not ising2 )


@pytest.mark.parametrize(
    "qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h, field, val_exp, vec_exp",
    [
        (
            "GridQubit",
            [2, 1],
            np.zeros((1, 1)),
            np.zeros((2, 0)),
            np.zeros((1, 1)),
            np.zeros((2, 0)),
            np.array([0.5, 1]).reshape((2,1)), 
            "X",
            [-0.75, -0.25],
            np.transpose([
                [-0.5, -0.5, -0.5, -0.5],
                [ 0.5,  0.5, -0.5, -0.5],
            ]),
        ),
        (
            "GridQubit",
            [2, 2],
            np.zeros((1, 2)),
            np.zeros((2, 1)),
            np.ones((1, 2)),
            np.ones((2, 1)),
            np.zeros((2, 2)),
            "X",
            [-1, -1],
            np.transpose([
                [1.+0.j , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0       ],
                [0      , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.+0.j  ],
            ]),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((1, 2)),
            np.ones((2, 1)),
            np.zeros((1, 2)),
            np.zeros((2, 1)),
            np.zeros((2, 2)),
            "X",
            [-1, -1],
            1/np.sqrt(8) * np.transpose([
                [ 1,  0, 0, -1,
                 0, -1, -1,  0,
                 0, -1, -1,  0,
                 -1, 0,  0, 1],
                [0, -1, -1, 0,
                 -1, 0, 0, 1,
                 -1, 0, 0, 1,
                 0, 1, 1, 0 ],
            ]),
        ),
        (
           "GridQubit",
            [2, 2],
            np.array([0, 0]).reshape((1,2)),
            np.array([0, 0]).reshape((2,1)),
            np.array([0.25, 0.75]).reshape((1,2)),
            np.array([0.5, 1]).reshape((2,1)),
            np.ones((2, 2)),
            "Z",
            [-1.625, -0.75],
            np.transpose([
                [1.+0.j , 0, 0, 0, 0, 0, 0, 0, 0     , 0, 0, 0, 0, 0, 0, 0],
                [0      , 0, 0, 0, 0, 0, 0, 0, 1.+0.j, 0, 0, 0, 0, 0, 0, 0],
            ]),
        ),
    ]
)

def test_diagonalise(qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h, field, val_exp, vec_exp):
    # Create IsingXY object
    np_sol           =  IsingXY(qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h, field)
    scipy_sol        =  IsingXY(qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h, field)
    sparse_scipy_sol =  IsingXY(qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h, field)
    
    #Calculate analytic results by different methods
    np_sol.diagonalise(solver = 'numpy')
    scipy_sol.diagonalise(solver = 'scipy')
    sparse_scipy_sol.diagonalise()
    
    # Test whether found eigenvalues are all close up to tolerance
    for i in range(2):
        compare_val_modulo_permutation(scipy_sol.eig_val, sparse_scipy_sol.eig_val, i)
        compare_val_modulo_permutation(np_sol.eig_val, sparse_scipy_sol.eig_val, i)
        compare_val_modulo_permutation(val_exp, sparse_scipy_sol.eig_val, i)
    
    # Test whether found eigenvectors are all close up to tolerance and global phase
    # Note that different eigen vectors can have a different global phase; hence we assert them one by one
    # Here we only check ground state and first excited state
    # Further issue: abitrary for degenerate
    for i in range(2):
        if np.abs(sparse_scipy_sol.eig_val[0] - sparse_scipy_sol.eig_val [1]) > 1e-14:
            #assert(sparse_scipy_sol.val[0] == sparse_scipy_sol.val[1] )
            compare_vec_modulo_permutation(scipy_sol.eig_vec , sparse_scipy_sol.eig_vec, i)
        
        compare_vec_modulo_permutation(np_sol.eig_vec, scipy_sol.eig_vec, i)
        compare_vec_modulo_permutation(vec_exp, scipy_sol.eig_vec, i)

def compare_val_modulo_permutation(A, B, i):
    try:
        np.testing.assert_allclose(A[i], B[i], rtol=1e-14, atol=1e-14)
    except AssertionError:
        np.testing.assert_allclose(A[(i+1)%2], B[i], rtol=1e-14, atol=1e-14)

def compare_vec_modulo_permutation(A, B, i):
    try:
        cirq.testing.lin_alg_utils.assert_allclose_up_to_global_phase(A[:,i], B[:,i], rtol=1e-14, atol=1e-14)
    except AssertionError:
        cirq.testing.lin_alg_utils.assert_allclose_up_to_global_phase(A[:,(i+1)%2], B[:,i], rtol=1e-14, atol=1e-14)

def test_json():
    model = IsingXY("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    
    json = model.to_json_dict()
    
    model2 = IsingXY.from_json_dict(json)
    
    assert (model == model2)

#############################################################
#                                                           #
#                    Assert tests                           #
#                                                           #
#############################################################
@pytest.mark.parametrize(
    "qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h",
    [
        (
            "GridQubit",
            [2, 2],
            np.ones((0, 2)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((0, 2)) / 2,
            np.ones((2, 2)) / 7,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((3, 2)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((3, 2)) / 3,
            np.ones((2, 2)) / 5,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 0)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((2, 0)) / 5,
            np.ones((2, 2)) / 5,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 3)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((2, 3)) / 3,
            np.ones((2, 2)) / 7,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2)) / 2,
            np.ones((3, 1)) / 5,
            np.ones((2, 2)) / 8,
            np.ones((3, 1)) / 2,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2)) / 2,
            np.ones((0, 1)) / 5,
            np.ones((2, 2)) / 4,
            np.ones((0, 1)) / 3,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2)) / 2,
            np.ones((2, 3)) / 5,
            np.ones((2, 2)) / 3,
            np.ones((2, 3)) / 4,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2)) / 2,
            np.ones((2, 0)) / 5,
            np.ones((2, 2)) / 3,
            np.ones((2, 0)) / 2,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((2, 2)) / 2,
            np.ones((2, 2)) / 2,
            np.ones((2, 1)),
        ),
    ],
)
def test_assert_set_jh(qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h):
    with pytest.raises(AssertionError):
        IsingXY(qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h)

@pytest.mark.parametrize(
    "qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h, field",
    [
        (
            "GridQubit",
            [2, 2],
            np.ones((0, 2)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((0, 2)) / 2,
            np.ones((2, 2)) / 7,
            np.ones((2, 2)),
            "blub"
        )]
)
def test_assert_field(qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h, field):
    with pytest.raises(AssertionError):
        IsingXY(qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h, field)

@pytest.mark.parametrize(
    "qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h",
    [
        (
            "GridQubit",
            [2, 2],
            np.ones((1, 2)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((1, 2)) / 2,
            np.ones((2, 2)) / 7,
            np.ones((2, 2))
        )]
)
def test_assert_energy(qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h):
    model = IsingXY(qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h)
    with pytest.raises(NotImplementedError):
        model.energy()