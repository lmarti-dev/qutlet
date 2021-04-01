#import required liberies:
#Possibly bad style to import external libaries here...
#..level for the moment until better solution
#.. import global to use them also in submodule
#https://stackoverflow.com/questions/11990556/how-to-make-global-imports-from-a-function

#imports within package

# subpackages
from fauvqe import (
    isings,
)

# Flattened module
from fauvqe.initialisers import (
    Initialiser,
)

# Flattened sub-modules
from fauvqe.isings import (
    Ising,
)

'''
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
'''