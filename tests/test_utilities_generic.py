# external imports
import pytest
import numpy as np
import cirq
import sympy

# internal imports
from fauvqe import (
    alternating_indices_to_sectors,
    commutator,
    direct_sum,
    flatten,
    flip_cross_rows,
    generalized_matmul,
    get_gate_count,
    greedy_grouping,
    hamming_weight,
    index_bits,
    interweave,
    merge_same_gates,
    orth_norm,
    ptrace,
    print_non_zero,
    sectors_to_alternating_indices,
)

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

"""
todo
def test_pi_matmul(l,correct):
    assert ut.pi_matmul(*l) == correct
def test_print_non_zero(M,correct):
    assert ut.print_non_zero(M) == correct
"""

@pytest.mark.parametrize(
    "M,correct,even_first",
    [
     ([0,1,2,3,4,5],[0,2,4,1,3,5],True), 
     ([[0,1,2,3,4,5],[0,1,2,3,4,5]],[[0,2,4,1,3,5],[0,2,4,1,3,5]],True), 
     ([[0,1,2,3,4,5],[6,7,8,9,10,11],[12,13,14,15,16,17],[18,19,20,21,22,23]],
            [[0,2,4,1,3,5],[12,14,16,13,15,17],[6,8,10,7,9,11],[18,20,22,19,21,23]],True), 
    ]
)

def test_alternating_indices_to_sectors(M,correct,even_first):
    assert (alternating_indices_to_sectors(np.array(M),even_first) == correct).all()

@pytest.mark.parametrize(
    "M,correct,even_first,axis",
    [
        ([0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5], True, 0),
        ([0, 2, 4, 1, 3, 5], [1, 0, 3, 2, 5, 4], False, None),
        (
            [[0, 2, 4, 1, 3, 5], [0, 2, 4, 1, 3, 5]],
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            True,
            (0, 1),
        ),
        (
            [
                [0, 2, 4, 1, 3, 5],
                [12, 14, 16, 13, 15, 17],
                [6, 8, 10, 7, 9, 11],
                [18, 20, 22, 19, 21, 23],
            ],
            [
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
                [12, 13, 14, 15, 16, 17],
                [18, 19, 20, 21, 22, 23],
            ],
            True,
            (0, 1),
        ),
    ],
)
def test_sectors_to_alternating_indices(M, correct, even_first, axis):
    assert (
        sectors_to_alternating_indices(np.array(M), even_first, axis)
        == np.array(correct)
    ).all()


@pytest.mark.parametrize(
    "M,correct,flip_odd",
    [
     (
        [[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5]]
        ,[[0,1,2,3,4,5],[5,4,3,2,1,0],[0,1,2,3,4,5]]
        ,True
     ),
    (
        [[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5]]
        ,[[0,1,2,3,4,5],[5,4,3,2,1,0],[0,1,2,3,4,5],[5,4,3,2,1,0]]
        ,True
    ),
    (
        [[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5]]
        ,[[5,4,3,2,1,0],[0,1,2,3,4,5],[5,4,3,2,1,0],[0,1,2,3,4,5]]
        ,False
    ),
    ]
)
def test_flip_cross_rows(M,correct,flip_odd):
    assert (flip_cross_rows(np.array(M),flip_odd)==correct).all()

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
    "a,correct",
    [
     (bin(300),[0,3,5,6]),
     (bin(399),[0,1,5,6,7,8]),
     (bin(1),[0]),
     (bin(0),[]),
     (bin(-29),[0,1,2,4]),
    ]
)
def test_indexbits(a,correct):
    assert index_bits(a)==correct

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