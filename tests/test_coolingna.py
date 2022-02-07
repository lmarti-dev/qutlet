# external imports
import pytest
import numpy as np
import cirq
from scipy.linalg import expm
import sympy

# internal imports
from fauvqe import Ising, CoolingNA, ExpectationValue

def test_copy():
    n = [1,3]; boundaries = [1, 0]
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])))
    m_anc = Ising("GridQubit", n, np.zeros((n[0]-boundaries[0], n[1])), np.zeros((n[0], n[1])), np.ones((n[0], n[1])))
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingNA(
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
    
    model = CoolingNA(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    model.set_circuit("qaoa")
    json = model.to_json_dict()
    
    model2 = CoolingNA.from_json_dict(json)
    
    assert (model == model2)

def test_energy_notimplemented():
    n = [1,2]; boundaries = [1, 0]
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1])), np.ones((n[0], n[1])))
    m_anc = Ising("GridQubit", n, np.zeros((n[0]-boundaries[0], n[1])), np.zeros((n[0], n[1])), np.ones((n[0], n[1])))
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingNA(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    model.set_circuit("qaoa")
    with pytest.raises(NotImplementedError):
        model.energy()