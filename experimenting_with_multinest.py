# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:27:30 2018

@author: isabeau
"""

import numpy as np
import random as rd
import pylab as pl
import matplotlib.pyplot as plt
from time import time
def generate_fake_data(n, m, c, sigma, x):
    d = np.zeros(n)
    for i in range (n):
        d[i] = m * x[i] + c + rd.normalvariate(0,sigma)
    
    return d

def straight_line(x, m, c):
    return m*x + c



#first step: generate the data
n   = 100
xmin = 0.
xmax = 100.
sigma = 0.5
stepsize = (xmax-xmin)/n
x = np.arange(xmin, xmax, stepsize)
d = generate_fake_data(n, 3.5, 1.2, sigma, x)
   
#plt.plot(x,d)
#plt.show() 
#second step: initiate multinest
from pymultinest.solve import Solver
from scipy.special import ndtri  

LN2PI = np.log(2.*np.pi)

class StraightLineModelPyMultiNest(Solver):
    """
    A simple straight line model, with a Gaussian likelihood.

    Args:
        data (:class:`numpy.ndarray`): an array containing the observed data
        abscissa (:class:`numpy.ndarray`): an array containing the points at which the data were taken
        modelfunc (function): a function defining the model
        sigma (float): the standard deviation of the noise in the data
        **kwargs: keyword arguments for the run method
    """

    # define the prior parameters
    cmin = -10.  # lower range on c (the same as the uniform c prior lower bound)
    cmax = 10.   # upper range on c (the same as the uniform c prior upper bound)

    mmu = 0.     # mean of the Gaussian prior on m
    msigma = 10. # standard deviation of the Gaussian prior on m

    def __init__(self, data, abscissa, modelfunc, sigma, **kwargs):
        # set the data
        self._data = data         # oberserved data
        self._abscissa = abscissa # points at which the observed data are taken
        self._sigma = sigma       # standard deviation(s) of the data
        self._logsigma = np.log(sigma) # log sigma here to save computations in the likelihood
        self._ndata = len(data)   # number of data points
        self._model = modelfunc   # model function

        Solver.__init__(self, **kwargs)

    def Prior(self, cube):
        """
        The prior transform going from the unit hypercube to the true parameters. This function
        has to be called "Prior".

        Args:
            cube (:class:`numpy.ndarray`): an array of values drawn from the unit hypercube

        Returns:
            :class:`numpy.ndarray`: an array of the transformed parameters
        """

        # extract values
        mprime = cube[0]
        cprime = cube[1]

        m = self.mmu + self.msigma*ndtri(mprime)      # convert back to m
        c = cprime*(self.cmax-self.cmin) + self.cmin  # convert back to c

        return np.array([m, c])

    def LogLikelihood(self, cube):
        """
        The log likelihood function. This function has to be called "LogLikelihood".

        Args:
            cube (:class:`numpy.ndarray`): an array of parameter values.

        Returns:
            float: the log likelihood value.
        """

        # extract parameters
        m = cube[0]
        c = cube[1]

        # calculate the model
        model = self._model(x, m, c)

        # normalisation
        norm = -0.5*self._ndata*LN2PI - self._ndata*self._logsigma

        # chi-squared
        chisq = np.sum(((self._data - model)/(self._sigma))**2)

        return norm - 0.5*chisq  
        
        
        
        
nlive = 1024 #number of live points
ndim = 2 #nember of parameters (n and c here)
tol = 0.5 #stopping criteria, smaller longer but more accurate

# run the algorithm
t0 = time()
solution = StraightLineModelPyMultiNest(d, x, straight_line, sigma, n_dims=ndim,
                                        n_live_points=nlive, evidence_tolerance=tol);
t1 = time()
print(t1-t0)

logZpymnest = solution.logZ #value of logZ the evidence
logZerrpymnest = solution.logZerr #estimate of the statistical uncertainty on logZ

print(logZpymnest)
print(logZerrpymnest)
print(solution)



mchain = solution.samples[:,0]

hist, bin_edges = np.histogram(mchain)
area = np.sum(mchain) * ( bin_edges[1] - bin_edges[0] )
cchain = solution.samples[:,1]
from scipy import stats
kernel = stats.gaussian_kde( mchain )

print( kernel.integrate_box_1d(3,4) )

m_values = np.arange(3.492,3.508,0.000001)
print(area)
plt.plot( m_values , kernel(m_values) * area )
plt.hist(mchain, bins=bin_edges)
#plt.ylim(0, 1000)
plt.show()
print(bin_edges)
#plt.hist(cchain)
#plt.show()



