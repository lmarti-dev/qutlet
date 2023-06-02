# external imports
import pytest
import numpy as np
import cirq
import sympy

# internal imports
from fauvqe import (
    commutator,
    direct_sum,
    flatten,
    get_gate_count,
    greedy_grouping,
    hamming_weight,
    merge_same_gates,
    orth_norm,
    ptrace,  
)

from fauvqe.utilities.generic import (
    alternating_indices_to_sectors,   
    arg_alternating_indices_to_sectors,
    arg_flip_cross_row,
    chained_matrix_multiplication,
    default_value_handler,
    flip_cross,
    flip_cross_cols,
    flip_cross_rows,
    generalized_matmul,
    grid_neighbour_list,
    grid_to_linear,
    interweave,
    index_bits,
    linear_to_grid,
    normalize_vec,
    sectors_to_alternating_indices,
    sum_divisible,
    sum_even,
    sum_odd,
    wrapping_slice,
)

from tests.test_helper_functions import do_lists_have_same_elements

@pytest.mark.parametrize(
    "test_circuit, gate_count",
    [
        (
            cirq.Circuit([cirq.Z(cirq.LineQubit(i)) for i in range(5)]),
            5
        ),
        (
             cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1)), cirq.H.on(cirq.GridQubit(0, 2)), 
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 1)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 2)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2))),
            8,
        ),
        (
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1)), cirq.H.on(cirq.GridQubit(0, 2)), 
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 0)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 1)),
                        (cirq.X**sympy.Symbol('b0')).on(cirq.GridQubit(0, 2)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        cirq.Moment(cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 0)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 1)),
                        cirq.PhasedXZGate(x_exponent= sympy.Symbol('x0'), 
                                        z_exponent= sympy.Symbol('z0'), 
                                        axis_phase_exponent= sympy.Symbol('a0')).on(cirq.GridQubit(0, 2))),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        cirq.FSimGate(sympy.Symbol('theta0'), sympy.Symbol('phi0')).\
                            on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2))),
            13,
        ),
        (
            cirq.Circuit(
                        (cirq.ZZ**(-0.665)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-0.665)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        (cirq.XX**(-0.665)).on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
                        (cirq.X**(-0.665)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-0.665)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-0.665)).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(-0.665)).on(cirq.GridQubit(3, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        cirq.reset(cirq.GridQubit(3,0)),
                        (cirq.ZZ**(-0.707)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-0.707)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        (cirq.XX**(-0.707)).on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
                        (cirq.X**(-0.707)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-0.707)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-0.707)).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(-0.707)).on(cirq.GridQubit(3, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        cirq.reset(cirq.GridQubit(3,0)),
                        ),
            18,
        ),
    ]
)
def test_get_gate_count(test_circuit, gate_count):
    assert get_gate_count(test_circuit) == gate_count

@pytest.mark.parametrize(
    "a,b,correct",
    [
        ([[1,2,3],[0,0,0],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],
           [[1,2,3,0,0,0],
           [0,0,0,0,0,0],
           [1,1,1,0,0,0],
           [0,0,0,1,1,1],
           [0,0,0,1,1,1],
           [0,0,0,1,1,1]]  
        ),
    ]
)
def test_direct_sum(a,b,correct):
    assert (direct_sum(np.array(a),np.array(b)) == np.array(correct)).all()

@pytest.mark.parametrize(
    "multiplication_function,test_args,ground_truth",
    [
        (
            np.kron,
            [[[1,2,3],[0,0,0],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]],
            np.array(
                [[1,1,1,2,2,2,3,3,3],
                [1,1,1,2,2,2,3,3,3],
                [1,1,1,2,2,2,3,3,3],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1]]
            )
        ),
        (
            direct_sum,
        [np.array([[1,2,3],[0,0,0],[1,1,1]]),
            np.array([[1,1,1],[1,1,1],[1,1,1]]),
            np.array([[2,2,2],[2,2,2],[3,3,3]])
        ],
           np.array(
               [[1,2,3,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [1,1,1,0,0,0,0,0,0],
                [0,0,0,1,1,1,0,0,0],
                [0,0,0,1,1,1,0,0,0],
                [0,0,0,1,1,1,0,0,0],
                [0,0,0,0,0,0,2,2,2],
                [0,0,0,0,0,0,2,2,2],
                [0,0,0,0,0,0,3,3,3]]
            )  
        ),
        (
            np.matmul,
            [   np.array([[3, 0], [0, 2]]), 
                np.array([[0, 2], [2, 0]]), 
                np.array([[1, 1], [1, 1]])], 
            np.array([[6, 6], [4, 4]]),
        ),
    ]
)
def test_generalized_matmul(multiplication_function,test_args,ground_truth):
    assert (generalized_matmul(multiplication_function,*test_args) == ground_truth).all()

@pytest.mark.parametrize(
    "a,correct",
    [
        ([[1,2,3],[],[4,[5,[6,7]],8,9]],[1,2,3,4,5,6,7,8,9]),
        ([[1,2,3],[4,[5,[6]]],7,8,[9]],[1,2,3,4,5,6,7,8,9]),
        ([[[],[1],[2,3]],[],[4,[5,6,7],8,9]],[1,2,3,4,5,6,7,8,9]),
        ([1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]),
    ]
)
def test_flatten(a,correct):
    assert list(flatten(a)) == correct

@pytest.mark.parametrize(
    "M,correct,even_first",
    [
        (
            [0,1,2,3,4,5],
            [0,2,4,1,3,5],
            True
        ), 
        (
            [[0,1,2,3,4,5],[0,1,2,3,4,5]],
            [[0,2,4,1,3,5],[0,2,4,1,3,5]],
            True
        ), 
        (
            np.array([[0,1,2,3,4,5],[6,7,8,9,10,11],[12,13,14,15,16,17],[18,19,20,21,22,23]]),
            [[0,2,4,1,3,5],[12,14,16,13,15,17],[6,8,10,7,9,11],[18,20,22,19,21,23]],
            True
            ), 
    ]
)
def test_alternating_indices_to_sectors(M,correct,even_first):
    assert (alternating_indices_to_sectors(M,even_first) == correct).all()

@pytest.mark.parametrize(
    "M,correct,even_first,axis",
    [
        (
            np.array([0, 2, 4, 1, 3, 5]), 
            np.array([0, 1, 2, 3, 4, 5]), 
            True, 
            0
        ),
        (
            np.array([0, 2, 4, 1, 3, 5]), 
            np.array([1, 0, 3, 2, 5, 4]),
            False, 
            None
        ),
        (
            np.array([[0, 2, 4, 1, 3, 5], [0, 2, 4, 1, 3, 5]]),
            np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]),
            True,
            (0, 1),
        ),
        (
            np.array([
                [0, 2, 4, 1, 3, 5],
                [12, 14, 16, 13, 15, 17],
                [6, 8, 10, 7, 9, 11],
                [18, 20, 22, 19, 21, 23],
            ]),
            np.array([
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
                [12, 13, 14, 15, 16, 17],
                [18, 19, 20, 21, 22, 23],
            ]),
            True,
            (0, 1),
        ),
    ],
)
def test_sectors_to_alternating_indices(M, correct, even_first, axis):
    assert (sectors_to_alternating_indices(M, even_first, axis)== correct).all()

@pytest.mark.parametrize(
    "i,correct",
    [
     (bin(24),2),
     (bin(-1),1),
     (bin(300),4),
     (bin(3),2),
     (199,5),
     (19,3),
     (bin(0),0),
     (0,0),
    ]
)
def test_hamming_weight(i,correct):
    assert hamming_weight(i)==correct

@pytest.mark.parametrize(
    "wrong_input",
    [
     (
        [0,1,2],
     ),
     (
        np.array([0,1,2]),
     ),
     (
        1.1
     )
    ]
)
def test_hamming_weight_error(wrong_input):
    with pytest.raises(TypeError):
        hamming_weight(wrong_input)

@pytest.mark.parametrize(
    "a,correct, ones",
    [
     (bin(300),[0,3,5,6], True),
     (bin(399),[0,1,5,6,7,8], True),
     (bin(1),[0], True),
     (bin(0),[], True),
     (bin(-29),[0,1,2,4], True),
     (10, [1, 3], False),
    ]
)
def test_index_bits(a,correct, ones):
    assert index_bits(a, ones)==correct

@pytest.mark.parametrize(
    "rho, ind, solution",
    [
        (
            1/4*np.eye(4),
            1,
            1/2*np.eye(2)
        ),
        (
            0.25*np.array(
                        [[1, 1, 0, 0], 
                        [1, 1, 0, 0], 
                        [0, 0, 1, 1], 
                        [0, 0, 1, 1]]
            ),
            0,
            0.5*np.array(
                        [[1, 1], 
                        [1, 1]]
            ),
        ),
        (
            1/8*np.eye(8),
            range(1, 3),
            1/2*np.eye(2)
        ),
        (
            0.5*np.array(
                        [[1, 0, 1, 0], 
                        [0, 0, 0, 0], 
                        [1, 0, 1, 0], 
                        [0, 0, 0, 0]]
            ),
            [1,],
            0.5*np.array(
                        [[1, 1], 
                        [1, 1]]
            ),
        ),
        (
            0.5*np.array(
                        [[1, 0, 1, 0], 
                        [0, 0, 0, 0], 
                        [1, 0, 1, 0], 
                        [0, 0, 0, 0]]
            ),
            [],
            0.5*np.array(
                        [[1, 0, 1, 0], 
                        [0, 0, 0, 0], 
                        [1, 0, 1, 0], 
                        [0, 0, 0, 0]]
            ),
        ),
    ]
)
def test_ptrace(rho, ind, solution):
    assert np.linalg.norm(ptrace(rho, ind) - solution) < 1e-7
    assert (ptrace(rho, ind) == solution).all()

def test_ptrace_errors():
    with pytest.raises(IndexError):
        ptrace(np.eye(2), 5)

@pytest.mark.parametrize(
    "A, B, solution",
    [
        (
            np.array(
                    [[1, 0], 
                    [0, -1]]
            ),
            np.array(
                    [[0, 1], 
                     [1, 0]]
            ),
            2j*np.array(
                    [[0, -1j], 
                    [1j, 0]]
            ),
        ),
        (
            np.eye(2),
            np.array(
                    [[1, 0], 
                    [0, -1]]
            ),
            0*np.eye(2)
        ),
    ]
)
def test_commutator(A, B, solution):
    assert np.linalg.norm(commutator(A, B) - solution) < 1e-7
    assert (commutator(A, B) == solution).all()

@pytest.mark.parametrize(
    "A, solution",
    [
        (
            np.array(
                    [[1, 0], 
                    [0, -1]]
            ),
            1
        ),
        (
            np.eye(2),
            0
        ),
    ]
)
def test_orth_norm(A, solution):
    assert abs(orth_norm(A) - solution) < 1e-7

@pytest.mark.parametrize(
    "pauli_sum, grouped_pauli_sum",
    [
        #
        #   1D test cases
        #
        (
            cirq.PauliSum.from_pauli_strings(
            0.5*cirq.Z.on(cirq.GridQubit(0,0))*cirq.Z.on(cirq.GridQubit(0,1))
            ),
            [cirq.PauliSum.from_pauli_strings(
            0.5*cirq.Z.on(cirq.GridQubit(0,0))*cirq.Z.on(cirq.GridQubit(0,1))
            )],
        ),
        (
            cirq.PauliSum.from_pauli_strings([
            1*cirq.Z.on(cirq.GridQubit(0,0))*cirq.Z.on(cirq.GridQubit(0,1)),
            2*cirq.Z.on(cirq.GridQubit(0,1))*cirq.Z.on(cirq.GridQubit(0,2)),
            ]),
            [
                cirq.PauliSum.from_pauli_strings(
                1*cirq.Z.on(cirq.GridQubit(0,0))*cirq.Z.on(cirq.GridQubit(0,1))
                ),
                cirq.PauliSum.from_pauli_strings(
                2*cirq.Z.on(cirq.GridQubit(0,1))*cirq.Z.on(cirq.GridQubit(0,2))
                ),
            ],
        ),
        (
            cirq.PauliSum.from_pauli_strings([
            1*cirq.Z.on(cirq.GridQubit(0,0))*cirq.Z.on(cirq.GridQubit(0,1)),
            2*cirq.Z.on(cirq.GridQubit(0,1))*cirq.Z.on(cirq.GridQubit(0,2)),
            3*cirq.Z.on(cirq.GridQubit(0,2))*cirq.Z.on(cirq.GridQubit(0,3)),
            ]),
            [
                cirq.PauliSum.from_pauli_strings([
                    1*cirq.Z.on(cirq.GridQubit(0,0))*cirq.Z.on(cirq.GridQubit(0,1)),
                    3*cirq.Z.on(cirq.GridQubit(0,2))*cirq.Z.on(cirq.GridQubit(0,3)),
                ]),
                cirq.PauliSum.from_pauli_strings(
                2*cirq.Z.on(cirq.GridQubit(0,1))*cirq.Z.on(cirq.GridQubit(0,2))
                ),
            ],
        ),
        (
            cirq.PauliSum.from_pauli_strings([
            1*cirq.Z.on(cirq.GridQubit(0,0))*cirq.Z.on(cirq.GridQubit(0,1)),
            2*cirq.Z.on(cirq.GridQubit(0,1))*cirq.Z.on(cirq.GridQubit(0,2)),
            3*cirq.Z.on(cirq.GridQubit(0,2))*cirq.Z.on(cirq.GridQubit(0,3)),
            0.3*cirq.Z.on(cirq.GridQubit(0,3))*cirq.Z.on(cirq.GridQubit(0,0)),
            ]),
            [
                cirq.PauliSum.from_pauli_strings([
                    1*cirq.Z.on(cirq.GridQubit(0,0))*cirq.Z.on(cirq.GridQubit(0,1)),
                    3*cirq.Z.on(cirq.GridQubit(0,2))*cirq.Z.on(cirq.GridQubit(0,3)),
                ]),
                cirq.PauliSum.from_pauli_strings([
                    2*cirq.Z.on(cirq.GridQubit(0,1))*cirq.Z.on(cirq.GridQubit(0,2)),
                    0.3*cirq.Z.on(cirq.GridQubit(0,3))*cirq.Z.on(cirq.GridQubit(0,0)),
                ]),
            ],
        ),
        #
        #   2D test cases
        #
        (
            cirq.PauliSum.from_pauli_strings([
            1*cirq.Z.on(cirq.GridQubit(0,0))*cirq.Z.on(cirq.GridQubit(0,1)),
            2*cirq.Z.on(cirq.GridQubit(0,1))*cirq.Z.on(cirq.GridQubit(0,2)),
            1.1*cirq.Z.on(cirq.GridQubit(1,0))*cirq.Z.on(cirq.GridQubit(1,1)),
            2.1*cirq.Z.on(cirq.GridQubit(1,1))*cirq.Z.on(cirq.GridQubit(1,2)),
            ]),
            [
                cirq.PauliSum.from_pauli_strings([
                    1*cirq.Z.on(cirq.GridQubit(0,0))*cirq.Z.on(cirq.GridQubit(0,1)),
                    1.1*cirq.Z.on(cirq.GridQubit(1,0))*cirq.Z.on(cirq.GridQubit(1,1)),
                ]),
                cirq.PauliSum.from_pauli_strings([
                    2*cirq.Z.on(cirq.GridQubit(0,1))*cirq.Z.on(cirq.GridQubit(0,2)),
                    2.1*cirq.Z.on(cirq.GridQubit(1,1))*cirq.Z.on(cirq.GridQubit(1,2)),
                ]),
            ],
        ),
    ]
)
def test_greedy_grouping(pauli_sum, grouped_pauli_sum):
    greedily_grouped_pauli_sum = greedy_grouping(pauli_sum)
    print(pauli_sum)
    print(grouped_pauli_sum)
    print(type(greedily_grouped_pauli_sum[0]))    
    assert greedily_grouped_pauli_sum == grouped_pauli_sum

@pytest.mark.parametrize(
    "init_circuit, final_circuit",
    [
        (
            cirq.Circuit(
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),cirq.H.on(cirq.GridQubit(2, 0)),
                    cirq.ZZ.on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    cirq.ZZ.on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                    cirq.ZZ.on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
            cirq.Circuit(
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),cirq.H.on(cirq.GridQubit(2, 0)),
                    cirq.ZZ.on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
            ),
        ),
        (
            cirq.Circuit(
                    (cirq.ZZ**(-0.333)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.333)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
            cirq.Circuit(
                    (cirq.ZZ**(-0.666)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
        ),
        (
            cirq.Circuit(
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.333)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.333)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
            cirq.Circuit(
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.666)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
        ),
        (
            cirq.Circuit(
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.333)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.334)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
            cirq.Circuit(
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.667)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
        ),
        (
            cirq.Circuit(
                    (cirq.ZZ**(-0.333)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.334)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
            cirq.Circuit(
                    (cirq.ZZ**(-0.667)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
        ),
        (
            cirq.Circuit(
                    (cirq.ZZ**(-0.333)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.Z.on(cirq.GridQubit(0, 0))),
                    (cirq.ZZ**(-0.334)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
            cirq.Circuit(
                    (cirq.Z.on(cirq.GridQubit(0, 0))),
                    (cirq.ZZ**(-0.667)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),   
            ),
        ),
        (
            cirq.Circuit(
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.111)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.222)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.334)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
            ),
            cirq.Circuit(
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.667)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
            ),
        ),
        (
            cirq.Circuit(
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.111)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.XX**(-0.222)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.334)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
            ),
            cirq.Circuit(
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
                    (cirq.XX**(-0.222)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.445)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
            ),
        ),
         (
            cirq.Circuit(
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.111)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                    (cirq.CNOT**(-0.222)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                    (cirq.ZZ**(-0.334)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
            ),
            cirq.Circuit(
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
                    (cirq.ZZ**(-0.111)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                    (cirq.CNOT**(-0.222)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                    (cirq.ZZ**(-0.334)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
            ),
        ),
        (
            cirq.Circuit(
                    (cirq.ZZ**(-0.333)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.Z.on(cirq.GridQubit(0, 0))),(cirq.X.on(cirq.GridQubit(1, 0))),
                    (cirq.ZZ**(-0.334)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
            cirq.Circuit(
                    (cirq.ZZ**(-0.333)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.Z.on(cirq.GridQubit(0, 0))),(cirq.X.on(cirq.GridQubit(1, 0))),
                    (cirq.ZZ**(-0.334)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
        ),
        (
            cirq.Circuit(
                    (cirq.CZ**(-0.333)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.Z.on(cirq.GridQubit(0, 0))),
                    (cirq.CZ**(-0.334)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
            cirq.Circuit(
                    (cirq.Z.on(cirq.GridQubit(0, 0))),
                    (cirq.CZ**(-0.667)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),    
            ),
        ),
        (
            cirq.Circuit(
                    (cirq.CZ**(-0.333)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.Z.on(cirq.GridQubit(1, 0))),
                    (cirq.CZ**(-0.334)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
            cirq.Circuit(
                    (cirq.Z.on(cirq.GridQubit(1, 0))),
                    (cirq.CZ**(-0.667)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ),
        ),
        (
            cirq.Circuit(
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
                    (cirq.CZ**(-0.333)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    (cirq.Z.on(cirq.GridQubit(1, 0))),
                    (cirq.CZ**(-0.334)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
            ),
            cirq.Circuit(
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
                    (cirq.Z.on(cirq.GridQubit(1, 0))),
                    (cirq.CZ**(-0.667)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                    cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)),
            ),
        ),
        #Note that here cirq.Z^2 is not dropped despite using
        # cirq.drop_negligible_operations
        (
            cirq.Circuit(
                        cirq.Y.on(cirq.GridQubit(0, 0)), cirq.Y.on(cirq.GridQubit(0, 1)),
                        (cirq.H**(-0.333)).on(cirq.GridQubit(0, 0)),
                        cirq.H.on(cirq.GridQubit(0, 1)),
                        (cirq.H**(-0.333)).on(cirq.GridQubit(0, 0)),
                        cirq.X.on(cirq.GridQubit(0, 1)),
                        cirq.Y.on(cirq.GridQubit(0, 0)), cirq.Y.on(cirq.GridQubit(0, 1)), 
                        ),
            cirq.Circuit(
                            cirq.Moment(cirq.Y(cirq.GridQubit(0, 0)),
                                        cirq.Y(cirq.GridQubit(0, 1)),
                                    ), 
                            cirq.Moment(cirq.H(cirq.GridQubit(0, 1)),
                                    ), 
                            cirq.Moment((cirq.H**-0.666).on(cirq.GridQubit(0, 0)),
                                        cirq.X(cirq.GridQubit(0, 1)),
                                    ), 
                            cirq.Moment(cirq.Y(cirq.GridQubit(0, 0)),
                                        cirq.Y(cirq.GridQubit(0, 1)),
                                    )   
            ),
        ),
    ]
)
def test_merge_same_gates(init_circuit, final_circuit):
    # Note that currently the merging happens from the end of the circuit
    # Effectively this moves 2 qubit gates towards the end and 1 qubit gates towards the beginning
    print("init_circuit:\n{}".format(init_circuit ))
    print("merge_same_gates(init_circuit):\n{}".format(merge_same_gates(init_circuit)))
    print("final_circuit:\n{}".format(final_circuit))
    assert merge_same_gates(init_circuit) == final_circuit

@pytest.mark.parametrize(
    "multiplication_rule,l,correct",
    [
        (
            np.kron,
            [[[1, 2, 3], [0, 0, 0], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
            [
                [1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 1, 1, 2, 2, 2, 3, 3, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
        ),
        (
            np.kron,
            [np.eye(2), np.ones((2, 2)), np.ones((2, 2))],
            np.kron(np.eye(2), np.ones((4, 4))),
        ),
        (
            direct_sum,
            [
                np.array([[1, 2, 3], [0, 0, 0], [1, 1, 1]]),
                np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                np.array([[2, 2, 2], [2, 2, 2], [3, 3, 3]]),
            ],
            np.array(
                [
                    [1, 2, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 2, 2, 2],
                    [0, 0, 0, 0, 0, 0, 2, 2, 2],
                    [0, 0, 0, 0, 0, 0, 3, 3, 3],
                ]
            ),
        ),
        (
            np.matmul,
            ([[3, 0], [0, 2]], [[0, 2], [2, 0]], [[1, 1], [1, 1]]),
            [[6, 6], [4, 4]],
        ),
    ],
)
def test_chained_matrix_multiplication(multiplication_rule, l, correct):
    assert (
        np.array(chained_matrix_multiplication(
                multiplication_rule, *l
            )
        )
        == np.array(correct)
    ).all()

@pytest.mark.parametrize(
    "M,correct,flip_odd",
    [
        (
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]],
            True,
        ),
        (
            [
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
            ],
            [
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
            ],
            True,
        ),
        (
            np.array(
                [
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                ]
            ),
            np.array(
                [
                    [5, 4, 3, 2, 1, 0],
                    [0, 1, 2, 3, 4, 5],
                    [5, 4, 3, 2, 1, 0],
                    [0, 1, 2, 3, 4, 5],
                ]
            ),
            False,
        ),
    ],
)
def test_flip_cross_rows(M, correct, flip_odd):
    assert (
        np.array(flip_cross_rows(M, flip_odd) == correct)
    ).all()

@pytest.mark.parametrize(
    "M,correct,rc,flip_odd",
    [
        (
            [[1, 2], [3, 4]],
            [[1, 2], [4, 3]],
            "r",
            True,
        ),
        (
            [[1, 2], [3, 4]],
            [[3, 2], [1, 4]],
            "c",
            False,
        ),
    ],
)
def test_flip_cross(M, correct, rc, flip_odd):
    assert (
        flip_cross(np.array(M), rc, flip_odd) == correct
    ).all()

@pytest.mark.parametrize(
    "M1,M2,correct",
    [
        (
            [1, 2, 3, 4], 
            [8, 9, 10, 11], 
            [1, 8, 2, 9, 3, 10, 4, 11],
        ),
        (
            [1, 2, 3, 4, 5], 
            [8, 9, 10, 11], 
            [1, 8, 2, 9, 3, 10, 4, 11, 5],
        ),
        #(
        #    [1, 2], 
        #    [8, 9, 10, 11], 
        #    [1, 8, 2, 9, 10, 11],
        #),
    ],
)
def test_interweave(M1, M2, correct):
    M1 = np.array(M1)
    M2 = np.array(M2)
    correct = np.array(correct)
    assert np.array(interweave(M1, M2) == correct).all()

@pytest.mark.parametrize(
    "indices,correct,N",
    [
        ((0, 1, 2, 3, 4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 3, 7), 8),
        ((3, 2), (6, 1), 10),
        ((0, 2), (0, 1), (2, 4)),
    ],
)
def test_arg_alternating_indices_to_sectors(indices, correct, N):
    assert np.array(
        arg_alternating_indices_to_sectors(
            indices=indices, N=N
        )
        == correct
    ).all()


def test_arg_alternating_indices_to_sectors_error():
    with pytest.raises(ValueError):
        arg_alternating_indices_to_sectors(
            indices=[1, 2],
            N=[
                2,
            ],
        )
    with pytest.raises(TypeError):
        arg_alternating_indices_to_sectors(
            indices=[1, 2], N="a"
        )


@pytest.mark.parametrize(
    "x,y,dimy,correct,flip_odd",
    [
        (2, 1, 4, (2, 1), True),
        (1, 1, 4, (1, 2), True),
        (1, 0, 4, (1, 3), True),
        (12, 1, 4, (12, 2), False),
        (11, 1, 4, (11, 1), False),
    ],
)
def test_arg_flip_cross_row(x, y, dimy, correct, flip_odd):
    assert (
        arg_flip_cross_row(
            x=x, y=y, dimy=dimy, flip_odd=flip_odd
        )
        == correct
    )


@pytest.mark.parametrize(
    "x,y,dimy",
    [
        (-2, 1, 4),
        (2, -1, 4),
        (2, 1, -4),
        (2, 10, 4),
    ],
)
def test_arg_flip_cross_row_error(x, y, dimy):
    with pytest.raises(ValueError):
        arg_flip_cross_row(x, y, dimy)


@pytest.mark.parametrize(
    "x,y,dimx,dimy,correct,horizontal",
    [
        (1, 2, 2, 4, 6, True),
        (0, 0, 2, 4, 0, True),
        (0, 3, 2, 4, 3, True),
        (2, 1, 4, 2, 5, True),
        (3, 1, 4, 2, 7, True),
        (4, 1, 4, 2, 8, False),
    ],
)
def test_grid_to_linear(x, y, dimx, dimy, correct, horizontal):
    assert (
        grid_to_linear(x, y, dimx, dimy, horizontal) == correct
    )


@pytest.mark.parametrize(
    "n,dimx,dimy,correct,horizontal",
    [
        (6, 2, 4, (1, 2), True),
        (6, 4, 2, (3, 0), True),
        (0, 10, 10, (0, 0), True),
        (6, 2, 4, (1, 2), True),
        (6, 4, 4, (2, 1), False),
    ],
)
def test_linear_to_grid(n, dimx, dimy, correct, horizontal):
    assert linear_to_grid(n, dimx, dimy, horizontal) == correct


@pytest.mark.parametrize(
    "v,correct",
    [((1, 1), np.array((1 / np.sqrt(2), 1 / np.sqrt(2)))), ((1, 0), (1, 0))],
)
def test_normalize_vec(v, correct):
    assert (normalize_vec(v) == correct).all()


@pytest.mark.parametrize(
    "l,i,correct",
    [
        ([2, 4, 6, 8], 2, 4),
        ([1, 3, 7, 9], 2, 0),
    ],
)
def test_sum_divisible(l, i, correct):
    assert sum_divisible(l, i) == correct


@pytest.mark.parametrize(
    "l,correct",
    [
        ([10, 20], 2),
        ([11, 21, 22], 1),
    ],
)
def test_sum_even(l, correct):
    assert sum_even(l) == correct


@pytest.mark.parametrize(
    "l,correct",
    [
        ([10, 20], 0),
        ([11, 21, 22], 2),
    ],
)
def test_sum_odd(l, correct):
    assert sum_odd(l) == correct


@pytest.mark.parametrize(
    "a,correct,ones",
    [
        (bin(300), [0, 3, 5, 6], True),
        (bin(399), [0, 1, 5, 6, 7, 8], True),
        (bin(1), [0], True),
        (bin(0), [], True),
        (bin(-29), [0, 1, 2, 4], True),
        (10, [1, 3], False),
    ],
)
def test_index_bits(a, correct, ones):
    assert index_bits(a, ones) == correct


@pytest.mark.parametrize(
    "i,shape,neighbour_order,periodic,diagonal,origin,correct",
    [
        (3, (4, 4), 1, True, True, "center", (0, 2, 3, 4, 6, 7, 12, 14, 15)),
        (2, (3, 6), 2, False, False, "topleft", (2, 3, 4, 8, 14)),
    ],
)
def test_grid_neighbour_list(
    i, shape, neighbour_order, periodic, diagonal, origin, correct
):
    neighbours = grid_neighbour_list(
        i, shape, neighbour_order, periodic, diagonal, origin
    )
    print(neighbours)
    assert do_lists_have_same_elements(neighbours, correct)

@pytest.mark.parametrize(
    "shape,value,correct",
    [
        ((3, 3), 3.3, np.full(shape=(3, 3), fill_value=3.3)),
        ((2, 2), "zeros", np.zeros(shape=(2, 2))),
        ((1, 1), "ones", np.ones(shape=(1, 1))),
        ((3, 3), "random", np.random.rand(3, 3)),
        ((10, 10), "zoink", None),
    ],
)
def test_default_value_handler(shape, value, correct):
    if value == "random":
        assert (
            default_value_handler(shape, value).shape
            == np.array(shape)
        ).all()
    elif value in ["zeros", "ones"] or isinstance(value, float):
        assert (
            default_value_handler(shape, value) == correct
        ).all()
    else:
        with pytest.raises(ValueError):
            default_value_handler(shape, value)


@pytest.mark.parametrize(
    "arr,indices,correct",
    [("abcde", [1, 2, 5, 6, 7], "bcabc"), ([10, 20, 30], [100, 200, 0], [20, 30, 10])],
)
def test_wrapping_slice(arr, indices, correct):
    assert wrapping_slice(arr, indices) == correct

def test_flip_cross_error():
    with pytest.raises(ValueError):
        flip_cross(M=[1, 2], rc="u")

@pytest.mark.parametrize(
    "M,correct,flip_odd",
    [
        (
            [[1, 2], [3, 4]],
            [[1, 4], [3, 2]],
            True,
        ),
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[3, 2], [1, 4]]),
            False,
        ),
    ],
)
def test_flip_cross_cols(M, correct, flip_odd):
    assert (
        np.array(flip_cross_cols(M, flip_odd) == correct)
    ).all()

