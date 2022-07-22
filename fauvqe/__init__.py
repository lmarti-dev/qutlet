# import required liberies:
# Possibly bad style to import external libaries here...
# ..level for the moment until better solution
# .. import global to use them also in submodule
# https://stackoverflow.com/questions/11990556/how-to-make-global-imports-from-a-function

# imports within package
from fauvqe.converter import (
    Converter
)

# subpackages
from fauvqe import models, objectives, optimisers

# Flattened sub-modules
from fauvqe.models import (
    AbstractModel,
    Adiabatic,
    DrivenModel,
    CooledAdiabatic,
    CoolingModel,
    Ising,
    IsingXY,
    Heisenberg,
    HeisenbergFC,
    SpinModel,
    SpinModelFC
)
from fauvqe.objectives import (
    AbstractExpectationValue,
    Correlation,
    CVaR,
    Entanglement,
    ExpectationValue,
    Fidelity,
    Magnetisation,
    MatrixCost,
    Objective,
    ObjectiveSum,
    TraceDistance,
    UtCost,
    Variance
)
from fauvqe.optimisers import (
    ADAM,
    GradientDescent,
    GradientOptimiser,
    OptimisationResult,
    OptimisationStep,
    Optimiser
)

from fauvqe.utilities  import (
    alternating_indices_to_sectors,
    check_type_and_convert,
    commutator,
    direct_sum,
    flatten,
    flatten_qubits,
    flip_cross_rows,
    haar,
    haar_1qubit,
    hamming_weight,
    index_bits,
    interweave,
    lists_almost_have_same_elements,
    lists_have_same_elements,
    niceprint,
    orth_norm,
    pi_direct_sum,
    pi_kron,
    pi_matmul,
    ptrace,
    print_non_zero,
    round_small_to_zero,
    sectors_to_alternating_indices,
    uniform,
    unitary_transpose,
)
"""
All within fauvqe imported libaries:
#Standard libaries
import random       # obvious
#import timeit       # to time run time
import collections  #for counters
import numpy as np  #obvious
import sympy        # sympy objects are needed to use variables in cirq circuits
#import pylikwid
#import os
import matplotlib.pyplot as plt   #to plot
from pprint import pprint

#Cirq libaries
import cirq
import qsimcirq
from cirqqulacs import QulacsSimulator
"""
