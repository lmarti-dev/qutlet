"""
    This Utilities submodule generates sets of random state vectors.

    Available is:
        -   Normalised Uniform Random
        -   1-qubit product state Haar-Random
        -   Fully Haar random

    Test question: who to vertify that some sample is drawn from a certain distribution
"""
import cirq
import joblib
import math
import numpy as np
import qsimcirq

def _get_haar_circuit(  n: int,
                        p: int): #-> cirq.Circuit()
    """
        Helper function to generate pseudo Haar random p-layer scrabling circuit for n qubits

                Parameters:
                        n           (int): Number of qubits
                        p           (int): depth of pseudorandom circuit

                Returns:
                        circuit (cirq.Circuit): Pseudorandom scrambling object as cirq Circuit
    """
    circuit = cirq.Circuit()
    _qubits=cirq.LineQubit.range(n)
    for i_p in range(p):
        circuit.append(_get_haar_1QubitLayer(_qubits))
        circuit.append(_get_haar_2QubitLayer(_qubits, i_p))

    return circuit

def _get_haar_1QubitLayer(_qubits):
    """
        Helper function to generate 1-qubite SU(2) layer of pseudo Haar random p-layer scrabling circuit for n qubits

                Parameters:
                        _qubits  (cirq.Linequbits?): cirq qubits in a list on which the gates should act on

                Returns:
                        cirq.MatrixGates generated from random SU(2) matrices
    """
    for qubit in _qubits:
        yield cirq.MatrixGate(  cirq.testing.random_unitary(2),
                                unitary_check_rtol=1e-12,
                                unitary_check_atol=1e-12,
                                ).on(qubit)

def _get_haar_2QubitLayer(_qubits, i_p):
    """
        Helper function to generate 2-qubite SU($) layer of pseudo Haar random p-layer scrabling circuit for n qubits

                Parameters:
                        _qubits  (cirq.Linequbits?): cirq qubits in a list on which the gates should act on

                Returns:
                        cirq.MatrixGates generated from random SU(4) matrices
    """
    for i_q in range(math.floor(len(_qubits)/2)):
        yield cirq.MatrixGate(  cirq.testing.random_unitary(4),
                                unitary_check_rtol=1e-12,
                                unitary_check_atol=1e-12,
                                ).on(   _qubits[2*i_q+np.mod(i_p,2)],
                                        _qubits[np.mod(2*i_q+1+np.mod(i_p,2), len(_qubits))])

    
# Note for the future maybe merge haar and haar_1qubit
def haar(n: int,
        m: int = 1,
        p: int = 20,
        reuse_circuit: bool = False,
        n_jobs: int = -1,
        simulator = None): #-> np.ndarray
    '''
        Generates m Haar random 2^n dim state vectors

                Parameters:
                        n           (int): Number of qubits
                        m           (int): Number of Haar random state vectors, default 1
                        p           (int): depth of pseudorandom circuit
                        n_jobs      (int): Number of joblib jobs to run in parallel, 
                                        default -1 to us joblib parallisation for n< 17 and qsim parallisation for n>16
                        simulator   (cirq.Simuator): Cirq Simulator object, default None, then use cirq Simulator with joblib and qsim otherwise

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
    ###############################################################################
    #TODO use joblib to generate m different circuits? 
    # or is it sufficent to use m different initial states? -> Need to test this
    # Note: For haar() using qsim or joblib barely makes a diiference 
    # Note: Reusing the circuit ruther does not pay off
    # Note: Using np.complex128 is nbarely slower
    if n_jobs == -1 and n > 16:
        n_jobs = 1
    else:
        n_jobs = 8

    if reuse_circuit:
        reuse_circuit= _get_haar_circuit(n,p)

    if n_jobs>1:
        if simulator is None:
            #Here using np.complex128 makes a lot of a difference in precision
            simulator = cirq.Simulator(dtype=np.complex128)
        m_rounded = math.ceil(m/n_jobs)*n_jobs
        random_states = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            joblib.delayed(_single_haar)(n, p, simulator, reuse_circuit)
            for j in range(m_rounded))
    else:
        if simulator is None:
            simulator = qsimcirq.QSimSimulator({"t": 8, "f": 4})
        random_states = []
        for i_m in range(m):
            random_states.append(_single_haar(n, p, simulator, reuse_circuit))

    #Renormalise all random state
    return np.array(random_states[0:m])*(1/np.linalg.norm(np.array(random_states[0:m]), axis=1)[:,None])

def _single_haar(   n: int,
                    p: int,
                    simulator,
                    haar_circuit = False):
    if haar_circuit is False:
        haar_circuit = _get_haar_circuit(n,p)
    rnd_int = np.random.randint(2**n)
    return simulator.simulate(  haar_circuit,
                                initial_state=rnd_int).state_vector()

def haar_1qubit(n: int,
                m: int = 1,
                n_jobs: int = -1,
                simulator = None): #-> np.ndarray
    '''
        Generates m Single Qubit Haar random product state vectors

                Parameters:
                        n           (int): Number of qubits
                        m           (int): Number of Haar random state vectors, default 1
                        n_jobs      (int): Number of joblib jobs to run in parallel, 
                                        default -1 to us joblib parallisation for n< 17 and qsim parallisation for n>16
                        simulator   (cirq.Simuator): Cirq Simulator object, default None, then use cirq Simulator with joblib and qsim otherwise

                Returns:
                        random_states (np.array): 2D numpy array of m Single Qubit Haar random product state vectors
    '''
    #TODO use joblib to generate m different circuits?
    # Round m to 8 devidible 
    if n_jobs == -1 and n > 16:
        n_jobs = 1
    else:
        n_jobs = 8

    if n_jobs>1:
        if simulator is None:
            simulator = cirq.Simulator(dtype=np.complex128)
        m_rounded = math.ceil(m/n_jobs)*n_jobs
        random_states = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            joblib.delayed(_single_haar_1qubit)(n, simulator)
            for j in range(m_rounded))
    else:
        if simulator is None:
            simulator = qsimcirq.QSimSimulator({"t": 8, "f": 4})
        random_states = []
        for i_m in range(m):
            random_states.append(_single_haar_1qubit(n, simulator))

    #Renormalise all random state
    return np.array(random_states[0:m])*(1/np.linalg.norm(np.array(random_states[0:m]), axis=1)[:,None])

def _single_haar_1qubit(n: int,
                        simulator):
    haar_1qubit_circuit = cirq.Circuit(
                            _get_haar_1QubitLayer(cirq.LineQubit.range(n))
    )
    rnd_int = np.random.randint(2**n)
    return simulator.simulate(  haar_1qubit_circuit,
                                initial_state=rnd_int).state_vector()

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
    #Renormalise all random state
    return np.array(random_states[0:m])*(1/np.linalg.norm(np.array(random_states[0:m]), axis=1)[:,None])