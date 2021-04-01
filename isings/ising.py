# %%
# external import
import numpy as np
import cirq
import importlib

# import all parent modules
from fauvqe.initialisers import Initialiser
# %%
class Ising(Initialiser):
    '''
    2D Ising class inherits initialiser 
    is mother of different quantum circuit methods
    '''
    #Import different quantum circuit algorithm into Ising class
    #increase readability as better structured
    #Issue with importing subpackage this way
    #altnative importlib.import_module('module')
    #qaoa = __import__('fauvqe.isings.circuits.qaoa')
    #it seems that this works:
    qaoa = importlib.import_module('fauvqe.isings.circuits.qaoa')
    #os = __import__('os')
    def __init__(self, qubittype, n, j_v, j_h, h):
        '''
            qubittype as defined in initialiser
            n number of qubits
            j_v vertical j's
            j_h horizontal j's
            h  external field
        '''
        # convert all input to np array to be sure
        super().__init__(qubittype, np.array(n)) 
        self._set_jh(j_v, j_h, h)
        super().set_simulator()

    def _set_jh(self, j_v, j_h, h):
        # convert input to numpy array to be sure
        j_v = np.array(j_v)
        #J vertical needs one row/horizontal line less
        #NEED FOR IMPROVEMENT
        assert(
            (j_v.shape == (self.n - np.array((1, 0)))).all() or 
            (j_v.shape == self.n).all()
        ), \
            "Error in Ising._set_jh(): j_v.shape != n - {{ (1,0), (0,0)}}, {} != {}".format(j_v.shape,(self.n - np.array((1, 0))))
        self.j_v = j_v;
        
        # convert input to numpy array to be sure
        j_h = np.array(j_h)
        #J horizontal needs one column/vertical line less#
        #NEED FOR IMPROVEMENT
        assert(
            (j_h.shape == (self.n - np.array((0, 1)))).all() or
            (j_h.shape == self.n ).all()
            ), \
            "Error in Ising._set_jh(): j_h.shape != n - {{ (0,1), (0,0)}}, {} != {}".format(j_h.shape,(self.n - np.array((0, 1))))
        self.j_h = j_h;

        #Set boundaries:
        self.boundaries = np.array(
            (self.n[0] - j_v.shape[0] , 
             self.n[1] - j_h.shape[1] ))

        # convert input to numpy array to be sure
        h = np.array(h)
        assert((h.shape ==  self.n).all()), \
            "Error in Ising._set_jh():: h.shape != n, {} != {}".format(h.shape,self.n)
        self.h = h;

    def energy(self, wf, field='X'):
        #maybe fuse with energy_JZZ_hZ partially somehow
        '''
        Energy for JZZ_hX Transverse field Ising model (TFIM) or JZZ-HZ Ising model

        Computes the energy-per-site of the Ising Model directly from the
        a given wavefunction. 
        Args:
            wf:     Array of size 2**(n[0]*n[1]) specifying the wavefunction.
                e.g.: wf2 = self.simulator.simulate(self.circuit).state_vector()

        Returns:
            energy: Float equal to the expectation value of the energy per site 
        
        Z is an array of shape (n_sites, 2**n_sites). Each row consists of the 
        2**n_sites non-zero entries in the operator that is the Pauli-Z matrix on
        one of the qubits times the identites on the other qubits. The
        (i*n_cols + j)th row corresponds to qubit (i,j).
        '''
        n_sites = self.n[0]*self.n[1]
        assert(2**n_sites == np.size(wf)),\
            "Error 2**n_sites != np.size(wf)"

        Z = np.array([(-1)**(np.arange(2**n_sites) >> i) for i in range(n_sites-1,-1,-1)])

        # Create the operator corresponding to the interaction energy summed over all
        # nearest-neighbor pairs of qubits
        ZZ_filter = np.zeros_like(wf, dtype=float)
        
        #Looping for soo many unnecessary ifs is bad.....
        #NEED FOR IMPROVEMENT - > avoid blank python for loops!!
        # 1. Sum over inner bounds
        # 2. Add possible periodic boundary terms
        # Do this to avoid if's with the loop
        #Previously:
        #for i in range(self.n[0]):
        #    for j in range(self.n[1]):
        #        if i < self.n[0]-self.boundaries[0]:
        #            ZZ_filter += self.j_v[i,j]*Z[i*self.n[1] + j]*Z[np.mod(i+1, self.n[0])*self.n[1] + j]
                    #ZZ_filter += self.j_v[i,j]*Z[i*self.n[1] + j]*Z[(i+1)*self.n[1] + j]
        #        if j < self.n[1]-self.boundaries[1]:
        #            ZZ_filter += self.j_h[i,j]*Z[i*self.n[1] + j]*Z[i*self.n[1] + np.mod(j+1, self.n[1])]
                    #ZZ_filter += self.j_h[i,j]*Z[i*self.n[1] + j]*Z[i*self.n[1] + (j+1)]
        
        # 1. Sum over inner bounds
        for i in range(self.n[0]-1):
            for j in range(self.n[1]-1):
                ZZ_filter += self.j_v[i,j]*Z[i*self.n[1] + j]*Z[(i+1)*self.n[1] + j]
                ZZ_filter += self.j_h[i,j]*Z[i*self.n[1] + j]*Z[i*self.n[1] + (j+1)]
        
        for i in range(self.n[0]-1):
            j = self.n[1] - 1
            ZZ_filter += self.j_v[i,j]*Z[i*self.n[1] + j]*Z[(i+1)*self.n[1] + j]

        for j in range(self.n[1]-1):
            i = self.n[0] - 1 
            ZZ_filter += self.j_h[i,j]*Z[i*self.n[1] + j]*Z[i*self.n[1] + (j+1)]
        
        # 2. Sum periodic boundaries
        if self.boundaries[1] == 0:
            for i in range(self.n[0]):
                j = self.n[1] - 1
                ZZ_filter += self.j_h[i,j]*Z[i*self.n[1] + j]*Z[i*self.n[1]]

        if self.boundaries[0] == 0:
            for j in range(self.n[1]):
                i = self.n[0] - 1 
                ZZ_filter += self.j_v[i,j]*Z[i*self.n[1] + j]*Z[j]
        

        # X and Z case for external field
        if field == 'X':
            # Expectation value of the energy divided by the number of sites
            #for hX need basis transform of wf 
            temp_circuit = cirq.Circuit()
            for row in self.qubits:
                for qubit in row:
                    temp_circuit.append(cirq.H(qubit))

            wf_x= self.simulator.simulate(temp_circuit,\
                    initial_state=wf).state_vector()
        
            # This might be WRONG!!!!
            #print("E(hX) = {}".format(sum(np.abs(wf_x)**2 * self.h.reshape(n_sites).dot(Z)) ))
            return np.sum(np.abs(wf)**2 * (-ZZ_filter) - np.abs(wf_x)**2 * self.h.reshape(n_sites).dot(Z) ) / n_sites

        elif field == 'Z':
            # Expectation value of the energy divided by the number of sites
            #energy_operator = -ZZ_filter + h.reshape(n_sites).dot(Z)
            #test show that sign in front of h needs to be +1, but not so obvious yet
            return np.sum(np.abs(wf)**2 * (-ZZ_filter + self.h.reshape(n_sites).dot(Z) )) / n_sites
        else:
            assert False, "Error in Ising.energy(): invalid external field basis, \n received: '{}', allowed is \
                'X'(default) and 'Z'".format(field)

    def set_circuit(self, qalgorithm, param, append = False):
        """
            Adds custom circuit to self.circuit (default)

            Args:
                qalgorithm : quantum algorithm option
                param:
                    hand over parameter to individual circuit method; e.g. qaoa
                    reset circuit (-> self. circuit = cirq. Circuit())

            Returns/Sets:
                circuit symp.Symbols array
                start parameters for circuit_parametrisation values; possibly at random or in call

            -AssertionError if circuit method does not exists
            -AssertionErrors for wrong parameter hand-over in individual circuit method itself.

            maybe use keyword arguments **parm later

            Need to generalise beta, gamma, beta_values, gamma_values to:

            obj.circuit_param           %these are the sympy.Symbols
            obj.circuit_param_values    %these are the sympy.Symbols values

            What to do with further circuit parameters like p?

            for qaoa want to call like:
                qaoa.set_symbols
                qaoa.set_beta_values etc...

            CHALLENGE: how to load class functions from sub-module?
        """
        if qalgorithm == 'qaoa':
            # set symbols gets as parameter QAOA repetitions p
            self.qaoa.set_symbols(self,param) 
            self.qaoa.set_circuit(self, append) #this is the former circuit_QAOA()
        else: 
            assert False, "Invalid quantum algorithm, received: '{}', allowed is \n \
                'qaoa'".format(qalgorithm)
    
    def set_circuit_param_values(self, new_values):
        assert(np.size(new_values) == np.size(self.circuit_param)),\
            "np.size(new_values) != np.size(self.circuit_param), {} != {}".format(np.size(new_values), np.size(self.circuit_param));
        self.circuit_param_values = new_values;

    def set_optimiser(self, optimiser_name, obj_func = 'X'):
        '''
            This function acts as an interface to the general optimiser structure
            Args:
                -optimiser_name : class name of an implemented optimiser
                -optional:
                    give the optimiser an alternative objective function other than
                    jZZ_hX energy
                    MISSING: give optimisers energy and change default field to 'Z'
                    https://stackoverflow.com/questions/38503937/parsing-default-arguments-in-functions-without-executing-the-them
        '''
        #Note obj_func = self.energy does not work as default!
        #NEED FOR IMPROVEMENT:
        if obj_func == 'X':
            obj_func = self.energy
            #print("X case")
        elif obj_func == 'Z':
            #Maybe make this with field 'X' to default
            obj_func = lambda input_wf: self.energy(input_wf, field='Z')
            #print("Z case")

        if optimiser_name == 'GradientDescent':
            #Import GradientDescent() class:
            gd = importlib.import_module('fauvqe.optimisers.gradient_descent')
            #WANT to pass all via view 
            #Maybe can do also somehow via cython and pointer??
            self.optimiser=gd.GradientDescent(
                obj_func, #note that this is a function + need to make more general \
                                    #Changed this from JZZ_hZ to JZZ_hX and no test failed
                                    #A) Tha's bad, B) needs to be more general. 
                                    # also want to give attributes within energy function
                self.qubits,
                self.simulator,
                self.circuit,
                self.circuit_param, 
                self.circuit_param_values.view(), #view() is ndarray method
            );
        else:
            assert False, "Invalid optimiser, received: '{}', allowed is \n \
                'GradientDescent'".format(optimiser_name)

    def get_spin_vm(self, wf):
        assert(np.size(self.n) == 2), "Expect 2D qubit grid"
       #probability from wf
        prob = abs(wf * np.conj(wf))

        #commulative probability
        n_temp = round(np.log2(wf.shape[0]))
        com_prob = np.zeros(n_temp)
        # now sum it
        #this is a potential openmp sum; loop over com_prob, use index arrays to select correct 
        
        for i in np.arange(n_temp):
            #com_prob[i] = sum wf over index array all in np
            #maybe write on 
            #does not quite work so do stupid version with for loop
            #declaring cpython types can maybe help, 
            #Bad due to nested for for if instead of numpy, but not straight forward
            for j in np.arange(2**n_temp):
                #np.binary_repr(3, width=4) use as mask
                if np.binary_repr(j, width=n_temp)[i] == '1':
                    com_prob[i] +=  prob[j]
        # This is for qubits:
        #{(i1, i2): com_prob[i2 + i1*q4.n[1]] for i1 in np.arange(q4.n[0]) for i2 in np.arange(q4.n[1])}
        #But we want for spins:
        return {(i0, i1): 2*com_prob[i1 + i0*self.n[1]]-1 for i0 in np.arange(self.n[0]) for i1 in np.arange(self.n[1])}

    def print_spin(self, wf):
        '''
            For cirq. heatmap see example:
            https://github.com/quantumlib/Cirq/blob/master/examples/bristlecone_heatmap_example.py
            value_map = {
                (qubit.row, qubit.col): np.random.random() for qubit in cirq.google.Bristlecone.qubits
            }
            heatmap = cirq.Heatmap(value_map)
            heatmap.plot()

            This is hard to test, but self.get_spin_vm(wf) is covered
            Possibly add test similar to cirq/vis/heatmap_test.py

            Further: add colour scale
        '''
        value_map = self.get_spin_vm(wf)
        # Create heatmap object
        heatmap = cirq.Heatmap(value_map)
        # Set colorscale to [-1, 1]
        heatmap.set_colormap(vmin= -1, vmax = +1)
        #Plot heatmap
        heatmap.plot()
    
    def energy_analytic_1d(self):
        '''
            Function that returns analytic solution for ground state energy
            of 1D TFIM as described inj Pfeuty, ANNALS OF PHYSICS: 57, 79-90 (1970) 
            Currently this ONLY WORKS FOR PERIODIC BOUNDARIES
        
            First assert if following conditions are met:
                -The given system is 1D
                -all h's have the same value
                -all J's have the same value, independant of which one is the
                    'used' direction

            Then:
                - Calculate \Lambda_k
                    For numeric reasons include h in \Lambda_k
                - Return E/N = - h* sum \Lambda_k/N
        '''
        assert(self.n[0]*self.n[1] == np.max(self.n)), \
                "Ising class error, given system dimensions n = {} are not 1D".format(self.n)
        assert(np.min(self.h) == np.max(self.h)), \
                "Ising class error, external field h = {} is not the same for all spins".format(self.h)
        #Use initial parameter to catch empty array
        assert((np.min(self.j_h,initial=np.finfo(np.float_).max)  == np.max(self.j_h,initial=np.finfo(np.float_).min) ) or (np.size(self.j_h) == 0)), \
                "Ising class error, interaction strength j_h = {} is not the same for all spins. max: {} , min: {}"\
                    .format(self.j_h, np.min(self.j_h,initial=np.finfo(np.float_).max) , np.max(self.j_h,initial=np.finfo(np.float_).min))
        #Use initial parameter to catch empty array
        assert((np.min(self.j_v,initial=np.finfo(np.float_).max)  == np.max(self.j_v,initial=np.finfo(np.float_).min) ) or (np.size(self.j_v) == 0)), \
                "Ising class error, interaction strength j_v = {} is not the same for all spins. max: {} , min: {}"\
                    .format(self.j_v, np.min(self.j_v,initial=np.finfo(np.float_).max) , np.max(self.j_v,initial=np.finfo(np.float_).min))
        lambda_k = self._get_lambda_k()
        print("#np.size(lambda_k) = {} \t self.n[0]*self.n[1] = {}"\
            .format(np.size(lambda_k), self.n[0]*self.n[1]))
        return -np.sum(lambda_k)/np.size(lambda_k) #self.n[0]*self.n[1]

    def _get_lambda_k(self):
        '''
            Helper function for energy_analytic_1d()
            Not intended for external call
        '''
        _n = self.n[0]*self.n[1]
        #print("_n: {}".format(_n))
        _k = 2*np.pi*np.arange(start=-(_n-np.mod(_n,2))/2, stop = _n/2+1E-10, step = 1 )/_n
        #print("_k: {}".format(_k))
        if self.j_h.size > 0:
            _j = self.j_h[0][0]
        else:
            _j = self.j_v[0][0]
        #print("_j: {}".format(_j))
        # Does not work for 0
        #_j = list(filter(None, (self.j_v[0], self.j_h[0])))
        return np.sqrt(self.h[0][0]**2 + _j**2 - (2*_j)*self.h[0][0]*np.cos(_k) )


#Add analytic energy function
#Add tests to test it
#Calculated by Pfeuty
#
#assert: 1D chain by either n 1D or product n vector = max n vector
#        J all same value
#        h all same value
#
#        Calc \Lambda_k
#        return E/N = - h/2 sum \Lambda_k


#--End of Class ising2d