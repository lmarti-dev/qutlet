# external imports
import pytest
import numpy as np
import cirq
from scipy.linalg import expm
import sympy

# internal imports
from fauvqe import Ising, CoolingModel, ExpectationValue, HeisenbergFC

@pytest.mark.parametrize(
    "n, boundaries",
    [
        ([1, 3], [1, 1]),
        ([1, 3], [1, 0]),
        ([3, 1], [1, 1]),
        ([3, 1], [0, 1]),
        ([3, 3], [0, 0]),
    ]
)
def test_copy(n, boundaries):
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])))
    m_anc = Ising("GridQubit", n, np.zeros((n[0]-boundaries[0], n[1])), np.zeros((n[0], n[1])), np.ones((n[0], n[1])))
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingModel(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    model.set_circuit("qaoa")
    model2 = model.copy()
    assert (model == model2)

@pytest.mark.parametrize(
    "n",
    [
        ([1, 3]),
        ([3, 1]),
        ([3, 3]),
    ]
)
def test_copy_fc(n):
    m_sys = HeisenbergFC("GridQubit", n, np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1], n[0], n[1])))
    m_anc = HeisenbergFC("GridQubit", n, np.zeros((n[0], n[1], n[0], n[1])), np.zeros((n[0], n[1], n[0], n[1])), np.zeros((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1])))
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingModel(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    model.set_circuit("qaoa")
    model2 = model.copy()
    assert (model == model2)


def test_json():
    n = [1,2]; boundaries = [1, 0]
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])))
    m_anc = Ising("GridQubit", n, np.zeros((n[0]-boundaries[0], n[1])), np.zeros((n[0], n[1])), np.ones((n[0], n[1])))
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingModel(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    model.set_circuit("qaoa")
    json = model.to_json_dict()
    
    model2 = CoolingModel.from_json_dict(json)
    
    assert (model == model2)

def test_energy_notimplemented():
    n = [1,2]; boundaries = [1, 0]
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])))
    m_anc = Ising("GridQubit", n, np.zeros((n[0]-boundaries[0], n[1])), np.zeros((n[0], n[1])), np.ones((n[0], n[1])))
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingModel(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    model.set_circuit("qaoa")
    with pytest.raises(NotImplementedError):
        model.energy()
    
def test_correct_ancilla_size():
    n = [1,2]; boundaries = [1, 0]
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])))
    m_anc = Ising("GridQubit", [1, 1])
    j_int = np.ones((1, n[0], n[1]))
    
    with pytest.raises(AssertionError):
        model = CoolingModel(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
        )