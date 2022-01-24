# external imports
import pytest
import numpy as np
import cirq
from scipy.linalg import expm
import sympy

# internal imports
from fauvqe import SpinModel

class MockSpinModel(SpinModel):
    def __init__(self):
        super().__init__("GridQubit", [1, 2], [np.ones((0, 2))], [np.ones((1, 1))], [np.ones((1, 2))], [lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)], [cirq.X])
    
    def copy(self):
        return MockSpinModel()

    def energy(self):
        return np.array([])

    def _set_hamiltonian(self, reset: bool = True):
        self.hamiltonian = cirq.PauliSum()
    
    @classmethod
    def from_json_dict(cls, dct):
        return MockSpinModel()

    def to_json_dict(self):
        return {}

def test_copy():
    model = MockSpinModel()
    model2 = model.copy()

    #Test whether the objects are the same
    assert( model == model2 )
    
    #But there ID is different
    assert( model is not model2 )

def test_json():
    model = MockSpinModel()
    
    json = model.to_json_dict()
    
    model2 = MockSpinModel.from_json_dict(json)
    
    assert (model == model2)