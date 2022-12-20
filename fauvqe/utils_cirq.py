import cirq
from typing import Tuple
import itertools
import fauvqe.utils as utils
import numpy as np
import openfermion as of
from fauvqe.models.abstractmodel import AbstractModel
import io
from typing import List

def qubits_shape(qubits: Tuple[cirq.Qid]):
    last_qubit = max(qubits)
    if isinstance(last_qubit,cirq.LineQubit):
        return (last_qubit.x+1,1)
    elif isinstance(last_qubit,cirq.GridQubit):
        return (last_qubit.row+1,last_qubit.col+1)

# shamelessly taken from stack
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


# Again this function comes from openfermion, I changed the eigenvalue function to a deterministic one so that I get the same ground state everytime in case it is degenerate
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

def pauli_str_is_hermitian(pstr:cirq.PauliString,anti:bool=False):
    if anti:
        return 1j*np.imag(pstr.coefficient) == pstr.coefficient
    else:
        return np.real(pstr.coefficient) == pstr.coefficient
def pauli_sum_is_hermitian(psum: cirq.PauliSum,anti:bool=False):
    return all(pauli_str_is_hermitian(pstr=pstr,anti=anti) for pstr in psum)


def make_pauli_str_hermitian(pstr,anti:bool=False):
    if pauli_str_is_hermitian(pstr=pstr,anti=not anti):
        return pstr.with_coefficient(1j*pstr.coefficient)
    if not anti:
        # hermitian A + A* = re(A)
        return pstr.with_coefficient(np.real(pstr.coefficient))
    else:
        # anti-hermitian A - A* = 1j*im(A)
        return pstr.with_coefficient(1j*np.imag(pstr.coefficient))
    
def make_pauli_sum_hermitian(psum: cirq.PauliSum,anti:bool=False):
    psum_out=cirq.PauliSum()
    if pauli_sum_is_hermitian(psum=psum,anti=anti):
        return psum
    psum_out = cirq.PauliSum.from_pauli_strings([make_pauli_str_hermitian(pstr) for pstr in psum])
    return psum_out


def qmap(model:AbstractModel):
    flattened_qubits = list(utils.flatten(model.qubits))
    return {k:v for k,v in zip(flattened_qubits,range(len(flattened_qubits)))}


def populate_empty_qubits(model: AbstractModel):
    circuit_qubits=list(model.circuit.all_qubits())
    model_qubits = model.flattened_qubits
    missing_qubits = [x for x in model_qubits if x not in circuit_qubits]
    circ = model.circuit.copy()
    if circuit_qubits==[]:
        print("The circuit has no qubits")
        
        circ = cirq.Circuit()
    circ.append([cirq.I(mq) for mq in missing_qubits])
    return circ

def match_param_values_to_symbols(model,symbols,default_value:str="zeros"):
    if model.circuit_param_values is None:
        model.circuit_param_values = np.array([])
    missing_size = np.size(symbols)-np.size(model.circuit_param_values)

    param_default_values = utils.default_value_handler(shape=(missing_size,),value=default_value)
    if missing_size > 0:
        model.circuit_param_values=np.concatenate((model.circuit_param_values,param_default_values))

def pauli_str_is_identity(pstr:cirq.PauliString) -> bool:
    if not isinstance(pstr,cirq.PauliString):
        raise ValueError("expected PauliString, got: {}".format(type(pstr)))
    return all(pstr.gate.pauli_mask == np.array([0]*len(pstr.gate.pauli_mask)).astype(np.uint8))

def all_pauli_str_commute(psum):
    for pstr1 in psum:
        for pstr2 in psum:
            if not cirq.commutes(pstr1,pstr2):
                return False
    return True
    
def pauli_string_exponentiation_circuit(pstr,theta):
    to_Z_basis=cirq.Circuit()
    from_Z_basis=cirq.Circuit()
    cnot_qs=[]
    for item in pstr.items():
        qubit,pauli = item
        
        if pauli == cirq.X:
            to_Z_basis.append(cirq.H(qubit))
            from_Z_basis.append(cirq.H(qubit))
        if pauli == cirq.Y:
            to_Z_basis.append(cirq.Rx(rads=np.pi/4).on(qubit))
            from_Z_basis.append(cirq.Rx(rads=-np.pi/4).on(qubit))
        if pauli != cirq.I:
            cnot_qs.append(qubit)
    qubits = list(pstr.keys())

    cnots = [cirq.CNOT(cnot_qs[n],cnot_qs[n+1]) for n in range(len(cnot_qs)-1)]
    cnot_circ = cirq.Circuit(*cnots)
    print(cnot_qs)
    circ = cirq.Circuit()
    
    circ.append(to_Z_basis)
    circ.append(cnot_circ)
    circ.append(cirq.Rz(rads=theta).on(qubits[-1]))
    circ.append(cirq.Circuit(reversed(cnot_circ)))
    circ.append(from_Z_basis)

    return circ

def even_excitation(coeff: float,indices: List[int],anti_hermitian:bool) -> of.FermionOperator:
    if len(indices)%2 !=0:
        raise ValueError("expected an even length for the indices list but got: {}".format(len(indices)))
    half_len = int(len(indices)/2)
    # split the indices between annihilation and creation parts
    iind = indices[:half_len]
    jind = indices[half_len:]
    # ai*aj*akal
    ac1 = ["{}^".format(n) for n in iind]
    aa1 = ["{}".format(n) for n in jind]
    # al*ak*ajai (h.c)
    ac2 = ["{}^".format(n) for n in jind]
    aa2 = ["{}".format(n) for n in iind]
    # partiy flip from resorting the operators
    if half_len == 1:
        parity = 1
    else:
        parity = (-1)**(half_len%2)

    op1=of.FermionOperator(" ".join(ac1 + aa1),coefficient=coeff)
    op2=parity*of.FermionOperator(" ".join(ac2 + aa2),coefficient=coeff)
    if anti_hermitian:
        return op1 - op2
    else:
        return op1 + op2

def single_excitation(coeff,i,j,anti_hermitian:bool):
    return even_excitation(coeff=coeff,indices=[i,j],anti_hermitian=anti_hermitian)

def double_excitation(coeff,i,j,k,l,anti_hermitian:bool):
    return even_excitation(coeff=coeff,indices=[i,j,k,l],anti_hermitian=anti_hermitian)
