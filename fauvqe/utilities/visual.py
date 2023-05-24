"""
    This Utilities submodule generates for visualisation

"""
#external imports
from cirq import Heatmap as cirq_Heatmap
import numpy as np

#internal imports
from fauvqe.models.abstractmodel import (
    AbstractModel,
)


def get_value_map_from_state(   model: AbstractModel, 
                                state: np.array,):
    assert np.size(model.n) == 2, "Expect 2D qubit grid"
    
    # probability from state
    probability = abs(np.vdot(state, state))

    # cumulative probability
    _n = round(np.log2(state.shape[0]))
    cumulative_probability = np.zeros(_n)

    # Now sum it; this is a potential openmp sum; loop over com_prob, use index arrays to select correct
    for i in np.arange(_n):
        for j in np.arange(2 ** _n):
            if np.binary_repr(j, width=_n)[i] == "1":
                cumulative_probability[i] += probability[j]
    
    # This is for qubits:
    # {(i1, i2): com_prob[i2 + i1*q4.n[1]] for i1 in np.arange(q4.n[0]) for i2 in np.arange(q4.n[1])}
    # But we want for spins:
    return {
        (i0, i1): 2 * cumulative_probability[i1 + i0 * model.n[1]] - 1
        for i0 in np.arange(model.n[0])
        for i1 in np.arange(model.n[1])
        }

def plot_heatmap(   model: AbstractModel, 
                    state: np.array,):
    """
    Currently does not work due to Cirq update...

    For cirq. heatmap see example:
    https://github.com/quantumlib/Cirq/blob/master/examples/bristlecone_heatmap_example.py
    https://github.com/quantumlib/Cirq/blob/master/examples/heatmaps.py
    https://github.com/quantumlib/Cirq/blob/master/cirq-core/cirq/vis/heatmap_test.py
    #https://quantumai.google/cirq/noise/heatmaps
    value_map = {
        (qubit.row, qubit.col): np.random.random() for qubit in cirq.google.Bristlecone.qubits
    }
    heatmap = cirq.Heatmap(value_map)
    heatmap.plot()

    This is hard to test, but value_map_from_state(wf) is covered
    Possibly add test similar to cirq/vis/heatmap_test.py

    Further: add colour scale
    """
    value_map = get_value_map_from_state(model, state)
    
    # Create heatmap object
    heatmap = cirq_Heatmap(value_map)

    # Plot heatmap
    heatmap.plot()

def print_non_zero( M,
                    name: str=None, 
                    eq_tol: float=1E-15):
    if name is not None:                # pragma: no cover 
        print(name)                     # pragma: no cover 
    print((abs(M)>eq_tol).astype(int))  # pragma: no cover 