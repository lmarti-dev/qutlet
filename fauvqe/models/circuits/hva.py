
from typing import Tuple
import cirq
from cirq.circuits import InsertStrategy

import openfermion as of
import sympy
from fauvqe.models.fermiHubbard import FermiHubbardModel
from fauvqe.models.fermionicModel import FermionicModel

import fauvqe.utils_cirq as cqutils
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

# it's the nth goshdarn time i redo this circuit. This time we go like z-string to s-string and then vertical hoppings along the whole thing.

def set_circuit(model: FermiHubbardModel,layers=1):
    if model.qubittype != "GridQubit":
        raise NotImplementedError("HVA not implemented for qubittype {}".format(model.qubittype))
    if model.encoding_options["encoding_name"] == "jordan_wigner":
        circuit=cirq.Circuit()
        symbols=[]
        # this circuit only works for models laid out like so:
        # u1 d1 u2 d2 
        # u3 d3 u4 d4
        rows = False # means precisely this because we do not weave rows but columns
        for layer in range(layers):
            layer_symbols=[]
            layer_symbols.append(sympy.Symbol("phi_{}".format(layer)))
            add_op_tree_as_circuit(main_circuit=circuit,
                                    op_tree=alternating_onsites(model,phi=layer_symbols[-1]),
                                    strategy=InsertStrategy.EARLIEST
                                )       
            add_op_tree_as_circuit(main_circuit=circuit,
                                    op_tree=weave_sectors(model=model,
                                                            rows=rows,
                                                            unweave=True,
                                                            fswap=True),
                                    strategy=InsertStrategy.EARLIEST
                                )       

            layer_symbols.append(sympy.Symbol("theta_h{}".format(layer)))
            # performs all horizontal hoppings (odd and even)
            add_op_tree_as_circuit(main_circuit=circuit,
                                    op_tree=horizontal_hoppings(model=model,
                                                        theta=layer_symbols[-1]),
                                    strategy=InsertStrategy.EARLIEST
                                )
            if model.n[0] > 1:
                # reverse odd rows with swaps to get from Z to S snake
                # and match even rows with FSWAPs to align corresponding indices
                add_op_tree_as_circuit(main_circuit=circuit,
                                            op_tree=ZtoS(model),
                                            strategy=InsertStrategy.EARLIEST
                                        )   
                # perform all vertical hoppings
                layer_symbols.append(sympy.Symbol("theta_v{}".format(layer)))
                for Np in range(int(model.n[1])):
                    add_op_tree_as_circuit(main_circuit=circuit,
                                            op_tree=Ul(model,split=False),
                                            strategy=InsertStrategy.EARLIEST
                                            )                    
                    add_op_tree_as_circuit(main_circuit=circuit,
                                            op_tree=Ur(model,split=False),
                                            strategy=InsertStrategy.EARLIEST
                                            )
                    add_op_tree_as_circuit(main_circuit=circuit,
                                            op_tree=vertical_hoppings(model=model,
                                                                theta=layer_symbols[-1],
                                                                split=False),
                                            strategy=InsertStrategy.EARLIEST
                                            )
                add_op_tree_as_circuit(main_circuit=circuit,
                            op_tree=ZtoS(model),
                            strategy=InsertStrategy.EARLIEST
                        )  
            add_op_tree_as_circuit(main_circuit=circuit,
                        op_tree=weave_sectors(model=model,
                                            rows=rows,
                                            unweave=False,
                                            fswap=True),
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
        raise NotImplementedError("No hva implementation for encoding: {}".format(model.encoding_options["encoding_name"]))        



def ZtoS(model,split:bool=False):
    op_tree=[]
    Nx,Ny = model.n
    for x in range(0,Nx):
        sub_op_tree=[]
        for y in range(Ny//2):
            fswap = bool(1-x%2)
            sub_op_tree.extend(f_or_swap_all_left(model,row_index=x,fswap=fswap,left=True))
            sub_op_tree.extend(f_or_swap_all_left(model,row_index=x,fswap=fswap,left=False))
        op_tree.extend(sub_op_tree)
    return op_tree    


def f_or_swap_all_left(model,row_index:int,left:bool,fswap:bool,split:bool=False):
    Nx,Ny = model.n
    il = -1 if left else 1
    al=int(left)
    op_tree=[]
    if fswap:
        qubit_op=of.FSWAP
    else:
        qubit_op=cirq.SWAP
    # start at 1 otherwise left and not left are the same
    for ny in range(1,Ny-(1-al),2):
            if ((left and ny == Ny/2) or (not left and ny == Ny/2-1)) and split:
                pass
            else:
                op_tree.append(qubit_op(model.qubits[row_index][ny],model.qubits[row_index][ny+il]))
    return op_tree
def add_op_tree_as_circuit(main_circuit,op_tree,strategy=InsertStrategy.NEW_THEN_INLINE):
    temp_circuit=cirq.Circuit()
    temp_circuit.append(op_tree,strategy=strategy)
    main_circuit.append(temp_circuit)

# weave up and down spin sectors
def weave_sectors(model,rows,unweave:bool=False,fswap:bool=False):
    op_tree=[]
    
    Nj,Ni = model.n
    if rows:
        Ni,Nj = Nj,Ni 
    for nj in range(1,Nj):
            sub_op_tree=[]
            for index in range(nj,Ni-nj,2):
                if fswap:
                    sub_op_tree.append(fswap_row_or_column(model=model,
                                                        index=index,
                                                        left=False,
                                                        row=rows))
                else:
                    sub_op_tree.append(swap_row_or_column(model=model,
                                                        index=index,
                                                        left=False,
                                                        row=rows))
            op_tree.append(sub_op_tree)
    if not unweave:
        op_tree = reversed(op_tree)
    return op_tree


def U_right_left(model,left: bool,fswap:bool=True,split:bool=True):
    op_tree=[]
    Nx,_ = model.n # Ny should ALWAYS be even because it's 2*y
    for nx in range(Nx):
        op_tree.append(f_or_swap_all_left(model,row_index=nx,left=left,fswap=fswap,split=split))
    return op_tree

def Ul(model,fswap:bool=True,split:bool=True): 
    return U_right_left(model=model,
                        left=True,
                        fswap=fswap,
                        split=split)
def Ur(model,fswap:bool=True,split:bool=True): 
    return U_right_left(model=model,
                        left=False,
                        fswap=fswap,
                        split=split)

def alternating_onsites(model,phi=0):
# applies onsite terms given that spins are alternating
    op_tree=[]
    Ni, Nj = model.n
    for ni in range(Ni):
        for nj in range(0,Nj-1,2):
            op_tree.append(onsite(model.qubits[ni][nj],model.qubits[ni][nj+1],phi=phi))
    return op_tree

def vertical_hoppings(model,theta=0,split:bool=True):
    # vertical hopping at the allowed bridges
    # if rows, the vertical hoppings are between columns, along the x-axis
    if split:
        return split_vertical_hoppings(model=model,theta=theta)
    else:
        return single_vertical_hoppings(model=model,theta=theta)
def single_vertical_hoppings(model,theta):
    op_tree=[]
    Nx,_=model.n
    for ni in range(0,Nx-1):
            # get the ventral pair of the outer pair of indices
            vertical_ind=ni%2-1
            op_tree.append(hopping(model.qubits[ni][vertical_ind],model.qubits[ni+1][vertical_ind],theta=theta))
    return op_tree
def split_vertical_hoppings(model,theta):
    op_tree=[]
    Ni, Nj = model.n
    inner_index = int(Nj/2) # should ALWAYS be int
    assert inner_index == Nj/2
    starts_inside = Ni%2
    for ni in range(0,Ni-1):
            # get the ventral pair of the outer pair of indices
            iii = starts_inside*inner_index 
            op_tree.append(hopping(model.qubits[ni][iii],model.qubits[ni+1][iii],theta=theta))
            op_tree.append(hopping(model.qubits[ni][iii-1],model.qubits[ni+1][iii-1],theta=theta))
            starts_inside = (starts_inside + 1)%2

    return op_tree

def horizontal_hoppings(model,theta=0):
    # if rows the horizontal hoppings are across rows, along the y-axis
    Ni, Nj = model.n
    op_tree=[]
    for odd_even in range(2):
        sub_op_tree=[]
        for ni in range(0,Ni):
            for nj in range(odd_even,Nj-1,2):
                # no hopping between up and down
                if nj == Nj/2-1 and bool(odd_even):
                    pass
                else:
                    sub_op_tree.append(hopping(model.qubits[ni][nj],model.qubits[ni][nj+1],theta=theta))
        op_tree.extend(sub_op_tree)
    return op_tree

class HoppingGate(cirq.Gate):
    def __init__(self, theta):
        super(HoppingGate, self)
        self.theta = theta

    def _num_qubits_(self):
        return 2

    def _unitary_(self):
        return np.array([
            [1,0,0,0],
            [0, np.cos(self.theta/2), -1j*np.sin(self.theta/2),0],
            [0,-1j*np.sin(self.theta/2), np.cos(self.theta/2),0],
            [0,0,0,1]
        ])

    def _circuit_diagram_info_(self, args):
        return f"Hop({self.theta})"

def hopping(qi: cirq.Qid, qj: cirq.Qid,theta=0):
    op_tree=[]
    # op_tree.append((cirq.Rz(rads=-np.pi/4).on(qi),
    #                 cirq.Rz(rads=np.pi/4).on(qj)))
    # op_tree.append(cirq.SQRT_ISWAP(qi,qj))
    # op_tree.append((cirq.Rz(rads=np.pi+theta/2).on(qi),
    #                 cirq.Rz(rads=-theta/2).on(qj)))
    # op_tree.append(cirq.SQRT_ISWAP(qi,qj))
    # op_tree.append((cirq.Rz(rads=np.pi*5/4).on(qi),
    #                 cirq.Rz(rads=-np.pi/4).on(qj)))
    # op_tree.append(HoppingGate(theta=theta).on(qi,qj))
    op_tree.append(cirq.ISwapPowGate(exponent=-2*theta).on(qi,qj))
    return op_tree


class OnsiteGate(cirq.Gate):
    def __init__(self, theta):
        super(OnsiteGate, self)
        self.theta = theta

    def _num_qubits_(self):
        return 2

    def _unitary_(self):
        return np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,np.exp(1j*self.theta)]
        ])

    def _circuit_diagram_info_(self, args):
        return f"Ons({self.theta})"



def onsite(qi: cirq.Qid, qj: cirq.Qid,phi=0):
    op_tree=[]
    # nu = sympy.asin(sympy.sqrt(2)*sympy.sin(phi/2))
    # xi = sympy.atan(sympy.tan(nu)/sympy.sqrt(2))
    # op_tree.append((cirq.Rz(rads=phi/2).on(qi),
    #                 cirq.Rz(rads=phi/2).on(qj)))
    # op_tree.append((cirq.Rx(rads=xi).on(qi),
    #                 cirq.Rx(rads=-np.pi/2).on(qj)))
    # op_tree.append(cirq.Z(qi))
    # op_tree.append(cirq.SQRT_ISWAP(qi,qj))
    # op_tree.append((cirq.Z(qi),
    #                 cirq.Rx(rads=-2*nu).on(qj)))
    # op_tree.append(cirq.SQRT_ISWAP(qi,qj))
    # op_tree.append((cirq.Rx(rads=xi).on(qi),
    #                 cirq.Rx(rads=np.pi/2).on(qj)))
    # op_tree.append(OnsiteGate(theta=phi).on(qi,qj))
    op_tree.append(cirq.FSimGate(theta=0,phi=phi).on(qi,qj))
    return op_tree

def hopswap(qi: cirq.Qid, qj: cirq.Qid):
    raise NotImplementedError

def fswap_row_or_column(model,index,left=True,row=True):
    return f_or_swap_row_or_column(model=model,
                                    index=index,
                                    left=left,
                                    row=row,
                                    fswap=True)

def swap_row_or_column(model,index,left=True,row=True):
    return f_or_swap_row_or_column(model=model,
                                    index=index,
                                    left=left,
                                    row=row,
                                    fswap=False)

def f_or_swap_row_or_column(model,index,left=True,row=True,fswap=True):
    (dimx,dimy) = cqutils.qubits_shape(model.flattened_qubits)
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
    (dimx,dimy) = cqutils.qubits_shape(model.flattened_qubits)
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

