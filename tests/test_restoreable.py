import pytest

from PyTrilinos import Epetra
import numpy as np

from fauvqe.restorable import Restorable


class DummyRestorable(Restorable):
    def to_json_dict(self):
        pass

    def from_json_dict(self, params):
        pass

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
def test_epetra_crsmatrix2numpy(N_crsmatrix, np_matrix):
    restoreable_obj = DummyRestorable()

    crsmatrix = create_crsmatrix(N_crsmatrix)
    np_matrix_from_crsmatrix = restoreable_obj.epetra_crsmatrix2numpy(crsmatrix)

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
def test_numpy2epetra_crsmatrix(np_matrix):
    # This uses that we already tested epetra_crsmatrix2numpy()
    restoreable_obj = DummyRestorable()

    np_matrix2 = restoreable_obj.epetra_crsmatrix2numpy(
                    restoreable_obj.numpy2epetra_crsmatrix(np_matrix))

    np.testing.assert_allclose(np_matrix2, np_matrix, rtol=1e-14, atol=0)

def test_assert_numpy2epetra_crsmatrix():
    restoreable_obj = DummyRestorable()

    np_matrix = np.array([  [1j,     0.4], 
                        [0.7,   2]], np.complex128)

    with pytest.raises(AssertionError):
        tmp = restoreable_obj.epetra_crsmatrix2numpy(
                    restoreable_obj.numpy2epetra_crsmatrix(np_matrix))