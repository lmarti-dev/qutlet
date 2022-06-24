#external imports:
import numpy as np
import pytest
from scipy import stats
from timeit import default_timer

# internal imports
from fauvqe import haar, haar_1qubit, uniform

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
def test_haar(n, m):
    random_states = haar(n,m)

    #Test whether shape is correct:
    assert ((m, 2**n) == np.shape(random_states))

    # Tes whether normalisation is correct
    probabilities = abs(random_states)**2
    assert (abs(np.sum(probabilities) - m) < 1e-14)

    #TODO test for distribution

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

@pytest.mark.parametrize(
    "n, m, n_jobs_slow",
    [
        (
            12, 16, 1
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

