import cirq
from typing import Tuple
import itertools
import fauvqe.utils as utils
import numpy as np
import openfermion as of
from fauvqe.models.abstractmodel import AbstractModel
import io

def qubits_shape(qubits: Tuple[cirq.Qid]):
    last_qubit = max(qubits)
    if isinstance(last_qubit,cirq.LineQubit):
        return last_qubit.x+1
    elif isinstance(last_qubit,cirq.GridQubit):
        return (last_qubit.row+1,last_qubit.col+1)


def depth(circuit: cirq.Circuit)-> int:
    depth = len(cirq.Circuit(circuit.all_operations()))
    return depth

def jw_spin_correct_indices(n_electrons, n_qubits):
    # since we usually fill even then odd indices, and in the default jw,
    # the up spins are even and odd are down, we check if we have the correct
    # up and down spin count before passing the indices back
    # when Nf is odd, we assume there is one more even indices, since those are filled even-first.
    combinations=itertools.combinations(range(n_qubits),n_electrons)
    correct_combinations=[list(c) for c in combinations if utils.sum_even(c)-utils.sum_odd(c)==n_electrons%2]
    jw_indices = [sum([2**iii for iii in combination]) for combination in correct_combinations]
    return jw_indices

# these are modified openfermion functions: they used a non-deterministic way (scipy.sparse.eigh) to find the gs which messes up
# any sort of meaningful comparison (when the gs is degenerate). This is slower, but deterministic.
# also implemented is the possibility to further restrict the Fock space to match spin occupations

def jw_spin_restrict_operator(sparse_operator,particle_number,n_qubits):
    if n_qubits is None:
        n_qubits = int(np.log2(sparse_operator.shape[0]))

    select_indices = jw_spin_correct_indices(n_electrons=particle_number,n_qubits=n_qubits)
    return sparse_operator[np.ix_(select_indices, select_indices)]


def eigenspectrum_at_particle_number(sparse_operator, particle_number,expanded=False,spin:bool=True):
    n_qubits = int(np.log2(sparse_operator.shape[0]))
    # Get the operator restricted to the subspace of the desired particle number
    if spin:
        jw_restrict_operator_func=jw_spin_restrict_operator
        jw_indices_func = jw_spin_correct_indices
    else:
        jw_restrict_operator_func=of.jw_number_restrict_operator
        jw_indices_func = of.jw_number_indices
    restricted_operator = jw_restrict_operator_func(sparse_operator=sparse_operator,
                                                    particle_number=particle_number,
                                                    n_qubits=n_qubits)
    # Compute eigenvalues and eigenvectors
    dense_restricted_operator = restricted_operator.toarray()
    eigvals, eigvecs = np.linalg.eigh(dense_restricted_operator)
    if expanded:
        expanded_eigvecs=np.zeros((2**n_qubits,2**n_qubits), dtype=complex)
        for iii in range(expanded_eigvecs.shape[-1]):
            expanded_eigvecs[jw_indices_func(n_electrons=particle_number, n_qubits=n_qubits),iii] = eigvecs[:,iii]
            return eigvals, expanded_eigvecs
    return eigvals, eigvecs
    
def jw_get_true_ground_state_at_particle_number(sparse_operator, particle_number,spin:bool=True):
    eigvals, eigvecs = eigenspectrum_at_particle_number(sparse_operator, particle_number,expanded=True,spin=spin)
    state = eigvecs[:, 0]
    return eigvals[0], state

def get_param_resolver(model: AbstractModel,param_values:np.ndarray=None):
    joined_dict = {**{str(model.circuit_param[i]): param_values[i] for i in range(len(model.circuit_param))}}
    return cirq.ParamResolver(joined_dict)

def svg_print_out(circuit,fpath="circuit"):
    utils.ensure_fpath(fpath=fpath)
    fout=io.open("{}.svg".format(fpath),"w+",encoding="utf8")
    fout.write(cirq.contrib.svg.circuit_to_svg(circuit))
    fout.close()

def make_pauli_sum_hermitian(psum,anti=False):
    psum_out=cirq.PauliSum()
    if not anti:
        for pstr in psum:
            psum_out += pstr.with_coefficient(np.real(pstr.coefficient))
    else:
        for pstr in psum:
            psum_out += pstr.with_coefficient(1j*np.imag(pstr.coefficient))
    return psum_out


def qmap(model:AbstractModel):
    return {k:v for k,v in zip(utils.flatten(model.qubits),range(len(utils.flatten(model.qubits))))}
