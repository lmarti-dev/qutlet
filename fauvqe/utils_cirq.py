import cirq
from typing import Tuple

def qubits_shape(qubits: Tuple[cirq.Qid]):
    last_qubit = max(qubits)
    if isinstance(last_qubit,cirq.LineQubit):
        return last_qubit.x+1
    elif isinstance(last_qubit,cirq.GridQubit):
        return (last_qubit.row+1,last_qubit.col+1)


def depth(circuit: cirq.Circuit)-> int:
    depth = len(cirq.Circuit(circuit.all_operations()))
    return depth