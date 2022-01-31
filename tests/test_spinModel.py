# external imports
import pytest
import numpy as np
import cirq
from scipy.linalg import expm
import sympy

# internal imports
from fauvqe import SpinModel

def test_copy():
    model = SpinModel("GridQubit", [1, 2], [np.ones((0, 2))], [np.ones((1, 1))], [np.ones((1, 2))], [lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)], [cirq.X])
    model.set_circuit("qaoa")
    model2 = model.copy()
    
    #Test whether the objects are the same
    assert( model == model2 )
    
    #But there ID is different
    assert( model is not model2 )

def test_json():
    model = SpinModel("GridQubit", [1, 2], [np.ones((0, 2))], [np.ones((1, 1))], [np.ones((1, 2))], [lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)], [cirq.X])
    
    json = model.to_json_dict()
    
    model2 = SpinModel.from_json_dict(json)
    
    assert (model == model2)