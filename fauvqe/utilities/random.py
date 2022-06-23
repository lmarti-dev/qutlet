"""
    This Utilities submodule generates sets of random state vectors.

    Available is:
        -   Normalised Uniform Random
        -   1-qubit product state Haar-Random
        -   Fully Haar random

    Test question: who to vertify that some sample is drawn from a certain distribution
"""
import cirq
import numpy as np
import qsimcirq

#def get_haar_circuit( int: n, int: p): #-> cirq.Circuit()
#    circuit = cirq.Circuit()
#
#    return circuit

def get_haar_1QubitLayer(n: int):
    pass

def get_haar_2QubitLayer(n: int):
    """
        s
    """
    pass

def haar(n: int,
        m: int = 1,
        p: int = 20): #-> np.ndarray
    '''
        Generates m Haar random 2^n dim state vectors

                Parameters:
                        n (int): Number of qubits
                        m (int): Number of Haar random state vectors, default 1
                        p (int): depth of pseudorandom circuit

                Returns:
                        random_states (np.array): 2D numpy array of m many Haar random 2^n state vectors
    '''
    # 1. Generate SU(2) decomposition of Haar random circuit as described in https://pennylane.ai/qml/demos/tutorial_haar_measure.html
    #       A. de Guise et al. 2017
    #       B. Recket al. 1994
    #       C. Clementset al. 2016
    #   Implement according to Clemente
    # Or rather: arXiv:math-ph/0609050v2  
    # 2. Generate m many random integers
    # 3. Use simulate_sweep to get m many Haar random 2^n dim state vectors
    # Test if it is Haarrandom based on eigenvalue density and spacing distribution
    # Circular Unitary Ensemble 
    # According to Clemente, depth should be 2^n???
    # Rather use arXiv:2203.16571v2  [quant-ph]  2 May 2022
    # Random quantum circuits are approximate unitary t-designs in depth O(nt^(5+o(1)))
    # t>2^n
    #OR Pseudo random states by Ji Liu Song
    # Test for Porter–Thomas distribution
    #Kolmogorov-Smirnov test
    # Resolution: implement 20 layer pseudorandom circuit
    haar_circuit = get_haar_circuit(n, p)
    rnd_initis = np.random.randint(m)

def haar_1qubit(n: int,
                m: int = 1): #-> np.ndarray
    pass

def uniform(n: int,
            m: int = 1): #-> np.ndarray
    '''
        Generates m Uniform random 2^n dim state vectors

                Parameters:
                        n (int): Number of qubits
                        m (int): Number of Uniform random state vectors, default 1

                Returns:
                        random_states (np.array): 2D numpy array of m many Uniform random 2^n state vectors
    '''
    random_states=2*np.random.rand(m,2**n).astype(np.complex128) - 1 + 2j*np.random.rand(m,2**n) - 1j
    norms = 1/np.linalg.norm(random_states, axis=1)
    random_states =np.squeeze(random_states)*norms[:,None]
    #random_states=np.squeeze(random_states)/np.linalg.norm(random_states, axis=1)
    #random_states=np.dot((1/np.linalg.norm(random_states, axis=1)),random_states)
    return random_states