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

from fauvqe import Converter


  
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
             cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.LineQubit(2*i))*cirq.Z(cirq.LineQubit(2*i+1)) for i in range(6)]) ,
             -np.array([np.sum(
                        np.prod(
                            (-2*np.array([int(i) for i in bin(int_in)[2:].zfill((2*6))])+1).reshape(int((2*6)/2), 2), 
                            axis=1)
                            ) for int_in in np.arange(2**(2*6))],
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

    #print(paulisum)
    #print(scipy_crsmatrix.diagonal())
    #This is to slow 
    np.testing.assert_allclose(true_diagonal, scipy_crsmatrix.diagonal(), rtol=1e-14, atol=0)


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
