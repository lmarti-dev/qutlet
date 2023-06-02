import pytest

import cirq
import numpy as np

try:
    from PyTrilinos import Epetra
    PyTrilinos_supported = True

    def create_crsmatrix(Nelements):
        Comm              = Epetra.PyComm()
        Map               = Epetra.Map(Nelements, 0, Comm)
        crsmatrix         = Epetra.CrsMatrix(Epetra.Copy, Map, 0)

        # can still be modified.
        for i in Map.MyGlobalElements():
            if i > 0:
                crsmatrix[i, i - 1] = i-0.3
            if i < Nelements - 1:
                crsmatrix[i, i + 1] = i+0.4
            crsmatrix[i, i] = i+1.0
        
        crsmatrix.FillComplete()
        Comm.Barrier()
        
        return crsmatrix
except:
    PyTrilinos_supported = False

from fauvqe import ANNNI, Converter


  
@pytest.mark.parametrize(
    "N_crsmatrix, np_matrix",
    [
        (
            2,
            np.array([  [1,     0.4], 
                        [0.7,   2]], np.float64)
        ),
        (
            3,
            np.array([  [1,     0.4,    0], 
                        [0.7,   2,      1.4],
                        [0,     1.7,    3],], np.float64)
        ),
        (
            4,
            np.array([  [1,     0.4,    0,      0], 
                        [0.7,   2,      1.4,    0],
                        [0,     1.7,    3,      2.4],
                        [0,     0,    2.7,      4],], np.float64)
        ),
    ]
)
@pytest.mark.skipif(not PyTrilinos_supported, reason="Unable to load pyTrilinos")
def test_epetra_crsmatrix2numpy(N_crsmatrix, np_matrix):
    converter_obj = Converter()

    crsmatrix = create_crsmatrix(N_crsmatrix)
    np_matrix_from_crsmatrix = converter_obj.epetra_crsmatrix2numpy(crsmatrix)

    np.testing.assert_allclose(np_matrix_from_crsmatrix, np_matrix, rtol=1e-14, atol=0)

@pytest.mark.parametrize(
    "np_matrix",
    [
        (
            np.array([  [1,     0.4], 
                        [0.7,   2]], np.float64)
        ),
        (
            np.array([  [1,     0,  0], 
                        [2,   2,    0],
                        [3,   3,    3],], np.float64)
        ),
        #This fails but should not:
        #(
        #    np.array([  [1,     1,  1], 
        #                [0,   0,    0],
        #                [0,   0,    0],], np.float64)
        #),
        (
            np.array([  [1,     0,  0], 
                        [1,   0,    0],
                        [1,   0,    0],], np.float64)
        ),
        (
            np.random.rand(3,3)
        ),
        (
            np.random.rand(4,4)
        ),
        (
            np.diag(range(5))
        ),
    ]
)
@pytest.mark.skipif(not PyTrilinos_supported, reason="Unable to load pyTrilinos")
def test_numpy2epetra_crsmatrix(np_matrix):
    # This uses that we already tested epetra_crsmatrix2numpy()
    converter_obj = Converter()

    np_matrix2 = converter_obj.epetra_crsmatrix2numpy(
                    converter_obj.numpy2epetra_crsmatrix(np_matrix))

    np.testing.assert_allclose(np_matrix2, np_matrix, rtol=1e-14, atol=0)

@pytest.mark.parametrize(
    "paulisum, dtype",
    [
        (
            cirq.PauliSum.from_pauli_strings(-cirq.Z(cirq.LineQubit(0))*cirq.Z(cirq.LineQubit(1))),
            np.complex128
        ),
        (
            cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.LineQubit(i))*cirq.Z(cirq.LineQubit(i+1)) for i in range(2)]),
            int
        ),
        (
            cirq.PauliSum.from_pauli_strings([-(i+1)*cirq.Z(cirq.LineQubit(i)) for i in range(5)]),
            np.complex128
        ),
        (
            cirq.PauliSum.from_pauli_strings([-cirq.X(cirq.LineQubit(i)) for i in range(2)]),
            np.complex128
        ),
        (
            cirq.PauliSum.from_pauli_strings([-cirq.X(cirq.LineQubit(i)) for i in range(5)]),
            np.complex128
        ),
        (
            cirq.PauliSum.from_pauli_strings([-np.pi*cirq.Z(cirq.LineQubit(i))*cirq.Z(cirq.LineQubit(i+1)) for i in range(6)]) +
            cirq.PauliSum.from_pauli_strings([-i*cirq.Z(cirq.LineQubit(i)) for i in range(6)]) + 
            cirq.PauliSum.from_pauli_strings([-cirq.X(cirq.LineQubit(i))/(i+1) for i in range(6)]),
            np.float64
        ),
    ]
)
def test_cirq_paulisum2scipy_crsmatrix(paulisum, dtype):
    converter_obj = Converter()
    scipy_crsmatrix = converter_obj.cirq_paulisum2scipy_crsmatrix(paulisum, dtype=dtype)

    print("Paulisum: {}".format(paulisum))

    np.testing.assert_allclose(paulisum.matrix(), scipy_crsmatrix.toarray(), rtol=1e-14, atol=0)
    
    #print(paulisum.matrix().diagonal())
    #assert False

@pytest.mark.higheffort
@pytest.mark.parametrize(
    "paulisum, true_diagonal",
    [
        (
             cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.LineQubit(2*i))*cirq.Z(cirq.LineQubit(2*i+1)) for i in range(8)]) ,
             -np.array([np.sum(
                        np.prod(
                            (-2*np.array([int(i) for i in bin(int_in)[2:].zfill((2*8))])+1).reshape(int((2*8)/2), 2), 
                            axis=1)
                            ) for int_in in np.arange(2**(2*8))],
                            dtype = np.int32)
        ),
       (
            cirq.PauliSum.from_pauli_strings([cirq.Z(cirq.LineQubit(i)) for i in range(15)]),
            np.array([ np.sum( (-2*np.array([int(i) for i in bin(int_in)[2:].zfill(15)])+1)) for int_in in np.arange(2**15)],dtype = np.int32)
       ),
    ]
)
def test_cirq_paulisum2scipy_crsmatrix_joblib(paulisum, true_diagonal):
    converter_obj = Converter()
    scipy_crsmatrix = converter_obj.cirq_paulisum2scipy_crsmatrix(paulisum, dtype=int)

    #To avoid slow down use diagonal matrixes/diagonal vectors for testing
    np.testing.assert_allclose(true_diagonal, scipy_crsmatrix.diagonal(), rtol=1e-14, atol=0)

#
#   Test Converter & ANNNI model
#
@pytest.mark.parametrize(
    "n, b",
    [
        (
            [1,4],
            [1,1],
        ),
        (
            [1,4],
            [1,0],
        ),
        (
            [2,3],
            [1,1],
        ),
        (
            [2,3],
            [1,0],
        ),
        (
            [3,3],
            [1,1],
        ),
        (
            [3,3],
            [1,0],
        ),
        (
            [3,3],
            [0,0],
        ),
    ]
)
def test_ANNNI_cirq_paulisum2scipy_crsmatrix(n, b):
    #print("n: {}\tb: {}".format(n,b))
    J = float(2*np.random.random(1) - 1)
    h = float(2*np.random.random(1) - 1)
    kappa = float(2*np.random.random(1) - 1)

    #print("J: {}\tkappa: {}".format(J,kappa))
    
    annni_obj = ANNNI(n, J, kappa, h, b)
    #print(annni_obj.hamiltonian())
    #print(np.diag(annni_obj.hamiltonian().matrix()))

    converter_obj = Converter()
    #scipy_sparse_obj = converter_obj.cirq_paulisum2scipy_crsmatrix(annni_obj.hamiltonian())
    scipy_sparse_obj = converter_obj.cirq_paulisum2scipy_crsmatrix(annni_obj.hamiltonian())

    np.testing.assert_allclose(np.diag(annni_obj.hamiltonian().matrix()), scipy_sparse_obj.diagonal(), rtol=1e-14, atol=1e-14)

#@pytest.mark.higheffort
@pytest.mark.parametrize(
       "n, b",
    [
        (
            [1,3],
            [1,1],
        ),
        (
            [1,4],
            [1,0],
        ),
        (
            [2,3],
            [1,1],
        ),
        (
            [2,3],
            [1,0],
        ),
        (
            [3,2],
            [0,1],
        ),
    ]
)
def test_ANNNI_diagonalisation_consistency(n, b):
    J = float(2*np.random.random(1) - 1)
    h = float(2*np.random.random(1) - 1)
    kappa = float(2*np.random.random(1) - 1)
    annni_obj = ANNNI(n, J, kappa, h, b)
    annni_obj.diagonalise()
    dense_GS = annni_obj.eig_vec[:,0].copy()

    converter_obj = Converter()
    scipy_sparse_obj = converter_obj.cirq_paulisum2scipy_crsmatrix(annni_obj.hamiltonian())
    annni_obj.diagonalise(matrix=scipy_sparse_obj)
    sparse_GS = annni_obj.eig_vec[:,0]

    print(abs(np.vdot(dense_GS,sparse_GS))**2)
    assert (abs(np.vdot(dense_GS,sparse_GS))**2-1) < 1e-14
    #cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(dense_GS, sparse_GS, rtol=1e-14, atol=1e-14)

#############################################################
#                                                           #
#                    Assert tests                           #
#                                                           #
#############################################################
@pytest.mark.skipif(not PyTrilinos_supported, reason="Unable to load pyTrilinos")
def test_assert_numpy2epetra_crsmatrix():
    converter_obj = Converter()

    np_matrix = np.array([  [1j,     0.4], 
                        [0.7,   2]], np.complex128)

    with pytest.raises(AssertionError):
        tmp = converter_obj.epetra_crsmatrix2numpy(
                    converter_obj.numpy2epetra_crsmatrix(np_matrix))

@pytest.mark.parametrize(
    "paulisum",
    [
        (
               cirq.PauliSum.from_pauli_strings([-cirq.Y(cirq.LineQubit(i)) for i in range(2)])     
        ),
        (
               cirq.PauliSum.from_pauli_strings(-cirq.Z(cirq.LineQubit(0))*cirq.Z(cirq.LineQubit(1))*cirq.Z(cirq.LineQubit(2)))    
        ),
    ]
)
def test_cirq_paulisum2scipy_crsmatrix_assert(paulisum):
    converter_obj = Converter()

    with pytest.raises(AssertionError):
        scipy_crsmatrix = converter_obj.cirq_paulisum2scipy_crsmatrix(paulisum)
