# import required liberies:
# Possibly bad style to import external libaries here...
# ..level for the moment until better solution
# .. import global to use them also in submodule
# https://stackoverflow.com/questions/11990556/how-to-make-global-imports-from-a-function
# subpackages
from fauvqe import models, objectives, optimisers

# Flattened sub-modules
from fauvqe.models import (
    AbstractModel,
    Adiabatic,
    ANNNI,
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
    commutator,
    Converter,
    direct_sum,
    flatten,
    flip_cross_rows,
    generalized_matmul,
    get_gate_count,
    greedy_grouping,
    haar,
    haar_1qubit,
    hamming_weight,
    index_bits,
    interweave,
    merge_same_gates,
    orth_norm,
    ptrace,
    print_non_zero,
    sectors_to_alternating_indices,
    uniform,
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
