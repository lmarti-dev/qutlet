from typing import Tuple
import cirq
from cirq.circuits import InsertStrategy

import openfermion as of
import sympy
from fauvqe.models.fermiHubbard import FermiHubbardModel
from fauvqe.models.fermionicModel import FermionicModel
import fauvqe.utils as utils
import numpy as np



"""
HAMILTONIAN VARIATIONAL ANSATZ
This file contains helper functions related to Fock models VQEs. 
They help implement the Hamiltonian Variational Ansatz (HVA) defined in Phys. Rev. A 92, 042303 (2015). and improved in arXiv:1912.06007
and discussed also in arXiv:2112.02025 [quant-ph] (we follow the last two)
"""

# if all this swap axis nonsense starts getting troublesome, swap the dimensions to force a vertical boundary in fermihubbard
# and remove all this useless code

# you cannot and I repeat CANNOT create a circuit and set it in another circuit with any other insert strategy than "new"
# thus parts of the circuits should not be converted to circuit.


def set_circuit(model: FermiHubbardModel,layers=1):
    if model.qubittype != "GridQubit":
        raise NotImplementedError("HVA not implemented for qubittype {}".format(model.qubittype))
    if model.encoding_name == "jordan_wigner":
        circuit=cirq.Circuit()
        # returns 0 if the longest edge is x(rows) else 1, if it is y(columns)
        # 0 if sectors are split by a vertical boundary and 1 if they are split by a horizontal boundary
        swap_axis = 0 if model.x_dimension >= model.y_dimension else 1
        # returns 0 if the vertical bridges are end-even, beginning-odd (for the first sector)
        # and returns 1 if thy are end-odd beginning-even
        vertical_bridges = 0 if max((model.x_dimension,model.y_dimension))%2==0 else 1
        symbols=[]

        for layer in range(layers):
            
            layer_symbols=[]
            layer_symbols.append(sympy.Symbol("theta_h{}".format(layer)))
            # performs all horizontal hoppings (odd and even)
            add_op_tree_as_circuit(main_circuit=circuit,
                                    op_tree=horizontal_hoppings(model=model,
                                                        swap_axis=swap_axis,
                                                        theta=layer_symbols[-1]),
                                    strategy=InsertStrategy.EARLIEST
                                )
            if min((model.x_dimension,model.y_dimension)) > 1:
                # perform all vertical hoppings
                n_minor=int(model.n[1-swap_axis])
                layer_symbols.append(sympy.Symbol("theta_v{}".format(layer)))
                for n_swaps in range(int(n_minor/2)):
                    #  do multiple only if you have more than 2 sites on the horizontal hop axis
                    if n_minor/2 > 2:
                        add_op_tree_as_circuit(main_circuit=circuit,
                                                op_tree=Ul(model=model,swap_axis=swap_axis),
                                                strategy=InsertStrategy.EARLIEST
                                        )
                    add_op_tree_as_circuit(main_circuit=circuit,
                                            op_tree=Ur(model=model,swap_axis=swap_axis),
                                            strategy=InsertStrategy.EARLIEST
                                        )
                    add_op_tree_as_circuit(main_circuit=circuit,
                                            op_tree=vertical_hoppings(model=model,
                                            swap_axis=swap_axis,
                                            vertical_bridges=vertical_bridges,
                                            theta=layer_symbols[-1]),
                                            strategy=InsertStrategy.EARLIEST
                                        )
            layer_symbols.append(sympy.Symbol("phi_{}".format(layer)))
            if min((model.x_dimension,model.y_dimension)) == 1:
                # perform only one onsite if you have a 1-by-N since ups and downs should already be neighbours
                add_op_tree_as_circuit(main_circuit=circuit,
                                        op_tree=alternating_onsites(model=model,swap_axis=swap_axis,phi=layer_symbols[-1]),
                                        strategy=InsertStrategy.EARLIEST
                                    )
            else:
                # put down spins next to up spins 
                add_op_tree_as_circuit(main_circuit=circuit,
                                        op_tree=spin_sectors_to_alternate(model=model,
                                                                            swap_axis=swap_axis,
                                                                            unweave=False),
                                        strategy=InsertStrategy.EARLIEST
                                    )
                # perform onsite
                add_op_tree_as_circuit(main_circuit=circuit,
                                        op_tree=alternating_onsites(model=model,swap_axis=swap_axis,phi=layer_symbols[-1]),
                                        strategy=InsertStrategy.EARLIEST
                                    )
                # put them back 
                add_op_tree_as_circuit(main_circuit=circuit,
                                        op_tree=spin_sectors_to_alternate(model=model,
                                                                            swap_axis=swap_axis,
                                                                            unweave=True),
                                        strategy=InsertStrategy.EARLIEST
                                   )
            symbols.append(layer_symbols)
        
        # give everything to the model
        model.circuit.append(circuit)
        model.circuit_param.extend(list(utils.flatten(symbols)))
        if model.circuit_param_values is None:
            model.circuit_param_values = np.array([])
        model.circuit_param_values=np.concatenate((model.circuit_param_values,np.zeros(np.size(symbols))))
    else:
        raise NotImplementedError("No hva implementation for encoding: {}".format(model.encoding_name))        


def add_op_tree_as_circuit(main_circuit,op_tree,strategy=InsertStrategy.NEW_THEN_INLINE):
    temp_circuit=cirq.Circuit()
    temp_circuit.append(op_tree,strategy=strategy)
    main_circuit.append(temp_circuit)

# weave up and down spin sectors
def spin_sectors_to_alternate(model,swap_axis,unweave=False):
    op_tree=[]
    n_minor=int(model.n[1-swap_axis])
    for nj in range(1,int(n_minor/2)):
            sub_op_tree=[]
            for index in range(nj,n_minor-nj,2):
                sub_op_tree.append(swap_row_or_column(model=model,
                                                    index=index,
                                                    left=False,
                                                    row=bool(swap_axis))
                                )
            op_tree.append(sub_op_tree)
    if not unweave:
        op_tree = reversed(op_tree)
    return op_tree


def U_right_left(model,swap_axis,left: bool):
    op_tree=[]
    n_minor=int(model.n[1-swap_axis])
    for nj in range(1,n_minor-(1-int(left)),2):
        # no fswapping up and down sector edges
        if (left and nj == n_minor/2) or (not left and nj == n_minor/2-1):
            pass
        else:
            op_tree.append(fswap_row_or_column(model=model,index=nj,left=left,row=bool(swap_axis)))
    return op_tree

def Ul(model,swap_axis): 
    return U_right_left(model=model,
                        swap_axis=swap_axis,
                        left=True)
def Ur(model,swap_axis): 
    return U_right_left(model=model,
                        swap_axis=swap_axis,
                        left=False)

def alternating_onsites(model,swap_axis,phi=0):
# applies onsite terms given that spins are alternating
    op_tree=[]
    n_minor=int(model.n[1-swap_axis])
    n_major=int(model.n[swap_axis])
    for ni in range(n_major):
        for nj in range(0,n_minor-1,2):
            if bool(swap_axis):
                op_tree.append(onsite(model.qubits[nj][ni],model.qubits[nj+1][ni],phi=phi))
            else:
                op_tree.append(onsite(model.qubits[ni][nj],model.qubits[ni][nj+1],phi=phi))
    return op_tree

def vertical_hoppings(model,swap_axis,vertical_bridges,theta=0):
    # vertical hopping at the allowed bridges
    op_tree=[]
    n_minor=int(model.n[swap_axis])
    n_major=int(model.n[swap_axis])
    for ni in range(0,n_major-1):
        inout = (ni + vertical_bridges) % 2
        if bool(1-swap_axis):
            if inout:
                op_tree.append(hopping(model.qubits[ni][0],model.qubits[ni+1][0],theta=theta))
                op_tree.append(hopping(model.qubits[ni][-1],model.qubits[ni+1][-1],theta=theta))
            else:
                op_tree.append(hopping(model.qubits[ni][int(n_minor/2-1)],model.qubits[ni+1][int(n_minor/2-1)],theta=theta))
                op_tree.append(hopping(model.qubits[ni][int(n_minor/2)],model.qubits[ni+1][int(n_minor/2)],theta=theta))
        elif bool(swap_axis):
            if inout:
                op_tree.append(hopping(model.qubits[0][ni],model.qubits[0][ni+1],theta=theta))
                op_tree.append(hopping(model.qubits[-1][ni],model.qubits[-1][ni+1],theta=theta))
            else:
                op_tree.append(hopping(model.qubits[int(n_minor/2-1)][ni],model.qubits[int(n_minor/2-1)][ni+1],theta=theta))
                op_tree.append(hopping(model.qubits[int(n_minor/2)][ni],model.qubits[int(n_minor/2)][ni+1],theta=theta))
    return op_tree

def horizontal_hoppings(model,swap_axis,theta=0):
    n_minor=int(model.n[1-swap_axis])
    n_major=int(model.n[swap_axis])
    op_tree=[]
    for odd_even in range(2):
        sub_op_tree=[]
        for ni in range(0,n_major):
            for nj in range(odd_even,n_minor-1,2):
                # no hopping between up and down
                if nj == n_minor/2-1 and bool(odd_even):
                    pass
                else:
                    if bool(1-swap_axis):
                        sub_op_tree.append(hopping(model.qubits[ni][nj],model.qubits[ni][nj+1],theta=theta))
                    else:
                        sub_op_tree.append(hopping(model.qubits[nj][ni],model.qubits[nj+1][ni],theta=theta))
        op_tree.extend(sub_op_tree)
    return op_tree

def hopping(qi: cirq.Qid, qj: cirq.Qid,theta=0):
    op_tree=[]
    op_tree.append((cirq.Rz(rads=-np.pi/4).on(qi),
                    cirq.Rz(rads=np.pi/4).on(qj)))
    op_tree.append(cirq.SQRT_ISWAP(qi,qj))
    op_tree.append((cirq.Rz(rads=np.pi+theta/2).on(qi),
                    cirq.Rz(rads=-theta/2).on(qj)))
    op_tree.append(cirq.SQRT_ISWAP(qi,qj))
    op_tree.append((cirq.Rz(rads=np.pi*5/4).on(qi),
                    cirq.Rz(rads=-np.pi/4).on(qj)))
    return op_tree

def onsite(qi: cirq.Qid, qj: cirq.Qid,phi=0):
    op_tree=[]
    nu = sympy.asin(sympy.sqrt(2)*sympy.sin(phi/2))
    xi = sympy.atan(sympy.tan(nu)/sympy.sqrt(2))
    op_tree.append((cirq.Rz(rads=phi/2).on(qi),
                    cirq.Rz(rads=phi/2).on(qj)))
    op_tree.append((cirq.Rx(rads=xi).on(qi),
                    cirq.Rx(rads=-np.pi/2).on(qj)))
    op_tree.append(cirq.Z(qi))
    op_tree.append(cirq.SQRT_ISWAP(qi,qj))
    op_tree.append((cirq.Z(qi),
                    cirq.Rx(rads=-2*nu).on(qj)))
    op_tree.append(cirq.SQRT_ISWAP(qi,qj))
    op_tree.append((cirq.Rx(rads=xi).on(qi),
                    cirq.Rx(rads=np.pi/2).on(qj)))
    return op_tree

def hopswap(qi: cirq.Qid, qj: cirq.Qid):
    raise NotImplementedError

def qubits_shape(qubits):
    last_qubit = max(qubits)
    if isinstance(last_qubit,cirq.LineQubit):
        return last_qubit.x+1
    elif isinstance(last_qubit,cirq.GridQubit):
        return (last_qubit.row+1,last_qubit.col+1)


def fswap_row_or_column(model,index,left,row):
    return f_or_swap_row_or_column(model=model,
                                    index=index,
                                    left=left,
                                    row=row,
                                    fswap=True)

def swap_row_or_column(model,index,left,row):
    return f_or_swap_row_or_column(model=model,
                                    index=index,
                                    left=left,
                                    row=row,
                                    fswap=False)

def f_or_swap_row_or_column(model,index,left=True,row=True,fswap=True):
    (dimx,dimy) = qubits_shape(model.flattened_qubits)
    if left and index < 1:
        raise ValueError("Cannot f/swap with left column when index is {}".format(index))
    dim_max = dimx if row else dimy
    if not left and index > dim_max-2:
        raise ValueError("Cannot f/swap with right column when index is {}".format(index))

    a=-1 if left else 1
    op_tree=[]
    qubit_op = of.FSWAP if fswap else cirq.SWAP
    #row means you swap between ROWS i.e. 0,0 with 1,0 and so on and 0,1 with 1,1
    if row:
        for y in range(dimy):
            op_tree.append(qubit_op(model.qubits[index][y],model.qubits[index+a][y]))
    else:
        for x in range(dimx):
            op_tree.append(qubit_op(model.qubits[x][index],model.qubits[x][index+a]))
    return op_tree


# don't use
def fswap_all_rows_or_columns(model,even=True,rows=True):
    (dimx,dimy) = qubits_shape(model.flattened_qubits)
    op_tree=[]
    if even:
        a=1
    else:
        a=2
    if rows:
        for index in range(a,dimx,2):
            op_tree += fswap_row_or_column(model=model,index=index,left=True,row=True)
    else:
        for index in range(a,dimy,2):
            op_tree += fswap_row_or_column(model=model,index=index,left=True,row=False)
    return cirq.Circuit(cirq.flatten_to_ops_or_moments(op_tree))

# do not use
def fswap_max(model: FermionicModel,horizontal=True,even=True):
    # if linequbit (presumably) raise error
    if model.qubittype != "GridQubit":
        raise NotImplementedError("This method only works for GridQubits")
    xdim,ydim = max(model.flattened_qubits).row+1,max(model.flattened_qubits).col+1
    
    op_tree=[]

    Z_mat=np.reshape(model.Z_snake,(xdim,ydim))
    if not horizontal:
        Z_mat = np.transpose(Z_mat)
        xdim,ydim = ydim,xdim
    for i in range(0,xdim):
        op_tree_o=[]
        op_tree_e=[]
        for start,op_tree_tmp in zip((0,1),(op_tree_e,op_tree_o)):
            for j in range(start,ydim-1,2): 
                if np.abs(Z_mat[i,j]-Z_mat[i,j+1])==1:
                    q1 = int(np.nonzero(model.Z_snake == Z_mat[i,j])[0])
                    q2 = int(np.nonzero(model.Z_snake == Z_mat[i,j+1])[0])
                    op_tree_tmp.append(of.FSWAP(
                                    model.flattened_qubits[q1],
                                    model.flattened_qubits[q2]
                                    )) 
        len_e=len(op_tree_e)  
        len_o=len(op_tree_o)
        e_is_longer = len_e > len_o
        oe_are_equal = len_e == len_o
        if  e_is_longer or (oe_are_equal and even):
            op_tree.extend(op_tree_e)
        elif not (e_is_longer or oe_are_equal) or (oe_are_equal and not even):
            op_tree.extend(op_tree_o)
    return cirq.Circuit(cirq.flatten_to_ops_or_moments(op_tree))

