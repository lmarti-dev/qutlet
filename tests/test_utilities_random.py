#external imports:
from cirq import Simulator as cirq_Simulator
import numpy as np
import pytest
from scipy import stats
from timeit import default_timer

# internal imports
from fauvqe import haar, haar_1qubit, uniform
from fauvqe.utilities.random import _single_haar

@pytest.mark.higheffort
@pytest.mark.parametrize(
    "n, m, atol, args",
    [
        (
            4, 15, 1e-13, {}
        ),
        (
            3, 23, 1e-13, {}
        ),
        (
            17, 1, 1e-6, {"reuse_circuit": True}
        ),
        (
            6, 1, 1e-6, {"n_jobs": 1}
        ),
    ]
)
def test_haar(n, m, atol, args):
    random_states = haar(n,m, **args)

    #Test whether shape is correct:
    assert ((m, 2**n) == np.shape(random_states))

    # Test whether normalisation is correct
    probabilities = abs(random_states)**2
    assert (abs(np.sum(probabilities) - m) < atol)

    #TODO test for distribution

def test__single_haar():
    random_state = _single_haar(10,10, cirq_Simulator(dtype=np.complex128))

    # Test whether normalisation is correct
    probabilities = abs(random_state)**2
    assert (abs(np.sum(probabilities) - 1) < 1e-13)

@pytest.mark.higheffort
@pytest.mark.parametrize(
    "n, m",
    [
        (
            4, 15
        ),
        (
            3, 23
        ),
    ]
)
def test_haar_1qubit(n, m):
    random_states = haar_1qubit(n,m)

    #Test whether shape is correct:
    assert ((m, 2**n) == np.shape(random_states))

    # Test whether normalisation is correct
    probabilities = abs(random_states)**2
    assert (abs(np.sum(probabilities) - m) < 1e-14)

    #TODO Add speed test -> new test 
    #TODO test for distribution

@pytest.mark.higheffort
@pytest.mark.parametrize(
    "n, m, n_jobs_slow",
    [
        (
            8, 128, 1
        ),
        (
            20, 32, 8
        ),
    ]
)
def test_haar_1qubit_speedup(n, m, n_jobs_slow):
    t0 = default_timer()
    random_states = haar_1qubit(n,m, n_jobs=n_jobs_slow)
    t1 = default_timer()
    random_states = haar_1qubit(n,m)
    t2 = default_timer()

    print("t1-t0: {}\tt2-t1 {}".format(t1-t0,t2-t1))
    assert ((t1-t0)>(t2-t1))

@pytest.mark.parametrize(
    "n, m",
    [
        (
            2, 100
        ),
        (
            3, 10
        ),
    ]
)
def test_uniform(n, m):
    # Test whether norms are 1
    # test for uniform distribution:
    # https://www.eg.bucknell.edu/~xmeng/Course/CS6337/Note/master/node43.html
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
    # Still need to figure good test
    
    random_states = uniform(n,m)

    #Test whether shape is correct:
    assert ((m, 2**n) == np.shape(random_states))

    probabilities = abs(random_states)**2
    #print(stats.kstest(probabilities, "uniform", N=m, alternative='two-sided', mode='auto'))
    #TODO test for distribution
    assert (abs(np.sum(probabilities) - m) < 1e-14)

