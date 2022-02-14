# external imports
import cirq
import pytest
import numpy as np
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


@pytest.mark.parametrize(
    "options_in, key, nested_level, options_out",
    [
        (
            {'a': 0, 'b': 1},
            'a',
            1,
            {'a': [0], 'b': 1}
        ),
        (
            {'a': 0, 'b': 1},
            'a',
            2,
            {'a': [[0]], 'b': 1}
        ),
        (
            {'a': {0}, 'b': 1},
            'a',
            1,
            {'a': [0], 'b': 1}
        ),
        (
            {'a': {0,2}, 'b': 1},
            'a',
            1,
            {'a': [0,2], 'b': 1}
        ),
        (
            {'a': {0,2}, 'b': 1},
            'a',
            2,
            {'a': [[0,2]], 'b': 1}
        ),
        (
            {'a': [0,2], 'b': 1},
            'a',
            1,
            {'a': [0,2], 'b': 1}
        ),
        (
            {'a': [[0,2]], 'b': 1},
            'a',
            2,
            {'a': [[0,2]], 'b': 1}
        ),
        #Flattening currently does not work
        #(
        #    {'a': [[0,2]], 'b': 1},
        #    'a',
        #    1,
        #    {'a': [0,2], 'b': 1}
        #),
    ]
)
def test__update2nestedlist(options_in, key, nested_level, options_out):
    model = SpinModel("GridQubit", [1, 2], [np.ones((0, 2))], [np.ones((1, 1))], [np.ones((1, 2))], [lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)], [cirq.X])
    options_in = model._update2nestedlist(options_in, key, nested_level)
    
    assert (options_in == options_out)


@pytest.mark.parametrize(
    "list, nested_level",
    [
        (
            5,
            0
        ),
        (
            'a',
            0
        ),
        (
            [0,2],
            1
        ),
        (
            [[0,2]],
            2
        ),
        (
            [[0,2], 3],
            2
        ),
        (
            [[[0]]],
            3
        ),
    ]
)
def test__nest_level(list, nested_level):
    model = SpinModel("GridQubit", [1, 2], [np.ones((0, 2))], [np.ones((1, 1))], [np.ones((1, 2))], [lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)], [cirq.X])

    assert (model._nest_level(list) == nested_level)