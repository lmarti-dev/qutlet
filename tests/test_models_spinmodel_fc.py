# external imports
import pytest
import numpy as np
import cirq
from scipy.linalg import expm
import sympy

# internal imports
from fauvqe import SpinModelFC

@pytest.mark.parametrize(
    "circuit",
    [
        ("qaoa"),("hea")
    ]
)
def test_copy(circuit):
    model = SpinModelFC("GridQubit", [1, 2], [np.ones((1, 2, 1, 2))], [np.ones((1, 2))], [lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)], [cirq.X])
    model.set_circuit(circuit)
    
    model2 = model.copy()
    if(hasattr(model, 'p')):
        model2.p = model.p
    
    #Test whether the objects are the same
    assert( model == model2 )
    
    #But there ID is different
    assert( model is not model2 )

def test_set_circuit_params():
    model = SpinModelFC("GridQubit", [1, 2], [np.ones((1, 2, 1, 2))], [np.ones((1, 2))], [lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)], [cirq.X])
    model.set_circuit("qaoa")
    model.set_circuit_param_values(np.ones(len(model.circuit_param)))
    assert True

def test_json():
    model = SpinModelFC("GridQubit", [1, 2], [np.ones((1, 2, 1, 2))], [np.ones((1, 2))], [lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)], [cirq.X])
    
    json = model.to_json_dict()
    
    model2 = SpinModelFC.from_json_dict(json)
    
    assert (model == model2)
    

def test_assert_set_circuit():
    model = SpinModelFC("GridQubit", [1, 2], [np.ones((1, 2, 1, 2))], [np.ones((1, 2))], [lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)], [cirq.X])
    with pytest.raises(AssertionError):
        model.set_circuit("blub")    