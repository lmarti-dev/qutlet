# external imports
import cirq
import pytest
import numpy as np
import scipy
from numbers import Real

# internal imports
from fauvqe import Ising, HeisenbergFC, MatrixCost, UtCost

@pytest.mark.parametrize(
    "n, boundaries, field, options, t",
    [
        (
            [2, 1], 
            [1, 1], 
            "X",
            {
                "trotter_order":1,
                "trotter_number":1,
            },
            0.3
        ),
        (
            [2, 1], 
            [1, 1], 
            "Z",
            {
                "trotter_order":1,
                "trotter_number":1,
            },
            0.3
        ),
        (
            [2, 1], 
            [1, 1], 
            "X",
            {
                "trotter_order":4,
                "trotter_number":1,
            },
            0.3
        ),
        (
            [2, 1], 
            [1, 1], 
            "X",
            {
                "trotter_order":1,
                "trotter_number":4,
            },
            1.0
        ),
    ]
)
def test_set_ising_trotter(n, boundaries, field, options, t):
    ising = Ising(  "GridQubit", 
                    n, 
                    np.ones((n[0]-boundaries[0], n[1])), 
                    np.ones((n[0], n[1]-boundaries[1])), 
                    np.ones((n[0], n[1])), 
                    field, 
                    t=t)
    ising.set_circuit("trotter", options)
    ising.set_simulator("cirq", {"dtype": np.complex128})
    cost = UtCost(ising, t)
    
    assert cost.evaluate(cost.simulate({})) < 1e-2

@pytest.mark.parametrize(
    "n, options, t",
    [
        (
            [2, 1], 
            {
                "trotter_order":1,
                "trotter_number":1,
            },
            0.3
        ),
        (
            [2, 1], 
            {
                "trotter_order":1,
                "trotter_number":1,
            },
            0.3
        ),
        (
            [2, 1], 
            {
                "trotter_order":4,
                "trotter_number":1,
            },
            0.3
        ),
        (
            [2, 1], 
            {
                "trotter_order":1,
                "trotter_number":4,
            },
            1.0
        ),
    ]
)
def test_set_heisenbergfc_trotter(n, options, t):
    heis = HeisenbergFC("GridQubit", n, np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1], n[0], n[1])), t=t)
    heis.set_circuit("trotter", options)
    heis.set_simulator("cirq", {"dtype": np.complex128})
    cost = UtCost(heis, t)
    
    assert cost.evaluate(cost.simulate({})) < 1e-2

@pytest.mark.parametrize(
    "n, boundaries, J, h, options, t, sol",
    [
        (
            [2, 1], 
            [1, 1], 
            1,
            2,
            {
                "trotter_order":1,
                "trotter_number":1,
            },
            0.3,
            (-2/np.pi)*0.3*np.array([1,2,2]),
        ),
        (
            [2, 1], 
            [1, 1], 
            1,
            2,
            {
                "trotter_order":1,
                "trotter_number":4,
            },
            0.3,
            1/4*(-2/np.pi)*0.3*np.array([1,2,2,1,2,2,1,2,2,1,2,2]),
        ),
        (
            [2, 1], 
            [1, 1], 
            1,
            2,
            {
                "trotter_order":2,
                "trotter_number":1,
            },
            1.0,
            0.5*(-2/np.pi)*1.0*np.array([1,2,2,2,2,1]),
        ),
    ]
)
def test_get_ising_trotter_params(n, boundaries, J, h, options, t, sol):
    ising = Ising("GridQubit", n, 
        J*np.ones((n[0]-boundaries[0], n[1])), 
        J*np.ones((n[0], n[1]-boundaries[1])), 
        h*np.ones((n[0], n[1])),
        t=t)
    ising.set_circuit("trotter", options)
    params = ising.trotter.get_parameters(ising)
    print(params)
    print(sol)
    assert (abs(params - sol) < 1e-13).all()

@pytest.mark.parametrize(
    "n, J, J2, options, t, sol",
    [
        (
            [2, 1], 
            1,
            2,
            {
                "trotter_order":1,
                "trotter_number":1,
            },
            0.3,
            (-2/np.pi)*0.3*np.array([1,1,2]),
        ),
        (
            [2, 1], 
            1,
            2,
            {
                "trotter_order":1,
                "trotter_number":4,
            },
            0.3,
            1/4*(-2/np.pi)*0.3*np.array([1,1,2,1,1,2,1,1,2,1,1,2]),
        ),
        (
            [2, 1], 
            1,
            2,
            {
                "trotter_order":2,
                "trotter_number":1,
            },
            1.0,
            0.5*(-2/np.pi)*np.array([1,1,2,2,1,1]),
        ),
    ]
)
def test_get_heisenbergfc_trotter_params(n, J, J2, options, t, sol):
    heis = HeisenbergFC("GridQubit", n, 
        J*np.ones((n[0], n[1], n[0], n[1])), 
        J*np.ones((n[0], n[1], n[0], n[1])), 
        J2*np.ones((n[0], n[1], n[0], n[1])),
        t=t
    )
    heis.set_circuit("trotter", options)
    params = heis.trotter.get_parameters(heis)
    print((params))
    print(sol)
    assert (abs(params - sol) < 1e-13).all()

@pytest.mark.parametrize(
    "n, boundaries, J, h, options, t",
    [
        (
            [2, 1], 
            [1, 1], 
            1,
            2,
            {
                "trotter_order":4,
                "trotter_number":1,
            },
            1.0,
        ),
    ]
)
def test_get_params_errors(n, boundaries, J, h, options, t):
    ising = Ising("GridQubit", n, 
        J*np.ones((n[0]-boundaries[0], n[1])), 
        J*np.ones((n[0], n[1]-boundaries[1])), 
        h*np.ones((n[0], n[1])),
        t=t)
    ising.set_circuit("trotter", options)
    with pytest.raises(NotImplementedError):
        ising.trotter.get_parameters(ising)

def _get_random_PauliSum(qubits, length, Isallpowgates):
    assert length >= len(qubits)-1
    #First do 2 qubit random paulis on all qubits
    #Then add more PauliStrings to get PauliSum length
    _SingleQubitGates=[cirq.X,cirq.Y,cirq.Z]
    if Isallpowgates:
        _TwoQubitGates=[[gate,gate] for gate in _SingleQubitGates]
    else:     
        _TwoQubitGates=[[gate1,gate2] for gate1 in _SingleQubitGates for gate2 in _SingleQubitGates]

    _paulisum = cirq.PauliSum()
    for i in range(len(qubits)-1):
        _rnd_int = np.random.randint(len(_TwoQubitGates), size=1)[0]
        _rnd_coefficent = float(2*np.random.rand(1)[0]-1)

        print(_TwoQubitGates[_rnd_int][0].on(qubits[i]))
        print(_TwoQubitGates[_rnd_int][1].on(qubits[i+1]) )
        _paulisum += (_rnd_coefficent*_TwoQubitGates[_rnd_int][0].on(qubits[i])*_TwoQubitGates[_rnd_int][1].on(qubits[i+1]) )

    for i in range(length+1-len(qubits)):
        _rnd_coefficent = float(2*np.random.rand(1)[0]-1)
        if float(2*np.random.rand(1)[0]-1) > 0:
            _rnd_int_gate = np.random.randint(len(_TwoQubitGates), size=1)[0]
            _rnd_int_qubit1 = np.random.randint(len(qubits), size=1)[0]
            _rnd_int_qubit2 = np.random.randint(len(qubits), size=1)[0]
            while _rnd_int_qubit1 == _rnd_int_qubit2:
                _rnd_int_qubit2 = np.random.randint(len(qubits), size=1)[0]
            _paulisum += (_rnd_coefficent   *_TwoQubitGates[_rnd_int_gate][0].on(qubits[_rnd_int_qubit1])\
                                            *_TwoQubitGates[_rnd_int_gate][1].on(qubits[_rnd_int_qubit2]) )
        else:
            _rnd_int_gate = np.random.randint(len(_SingleQubitGates), size=1)[0]
            _rnd_int_qubit = np.random.randint(len(qubits), size=1)[0]
            _paulisum += (_rnd_coefficent*_SingleQubitGates[_rnd_int_gate].on(qubits[_rnd_int_qubit]) )

    return _paulisum
        
def _previous_circuit(  hamiltonian: cirq.PauliSum, 
                        t: Real):
    res = cirq.Circuit()
    #Loop through all the addends in the PauliSum Hamiltonian
    for pstr in hamiltonian._linear_dict:
        #temp encodes each of the PauliStrings in the PauliSum hamiltonian which need to be turned into gates
        temp = 1
        #Loop through Paulis in the PauliString (pauli[1] encodes the cirq gate and pauli[0] encodes the qubit on which the gate acts)
        for pauli in pstr:
            temp = temp * pauli[1](pauli[0])
        #   Append the PauliString gate in temp to the power of the time step * coefficient of said PauliString. 
        # The coefficient needs to be multiplied by a correction factor of 2/pi in order for the PowerGate to represent a Pauli exponential.

        res.append(temp**np.real(2/np.pi * float(t) * float(np.real(hamiltonian._linear_dict[pstr]))))

        #Copy the Trotter layer *m times.
    return res

@pytest.mark.parametrize(
    "n, string_length, Isallpowgates",
    [
        (
            2,
            1,
            True,      
        ),
        (
             2,
             5,
             True,       
        ),
        (
             2,
             3,
             False,        
         ),
         (
            3,
            2,
            True,      
        ),
        (
             3,
             5,
             True,       
        ),
        (
             3,
             3,
             False,        
         ),
        (
            4,
            3,
            True,       
        ),
        (
            4,
            5,
            True,     
        ),
        (
            4,
            3,
            False,       
        ),
    ]
)
def test_consistency_previous_circuit_implementation(n, string_length, Isallpowgates):
    t = 1 + float(np.random.rand(1)[0])
    qubits=cirq.LineQubit.range(n)

    #get random PauliString on n qubits
    #with/without non-PowGates
    _paulisum = _get_random_PauliSum(qubits, string_length, Isallpowgates)

    _old_circuit=_previous_circuit(_paulisum, t)

    dummy_model = Ising(  "GridQubit", 
                    [1,n], 
                    np.ones((0, n)), 
                    np.ones((1, n-1)), 
                    np.ones((1, n)), 
                    "X", 
                    t=t)

    _new_circuit=dummy_model.trotter._first_order_trotter_circuit(   dummy_model, _paulisum, t)

    cost=MatrixCost(dummy_model, cirq.unitary(_new_circuit), exponent=2)

    assert abs(cost.evaluate(cirq.unitary(_old_circuit))) < 1e-14
    