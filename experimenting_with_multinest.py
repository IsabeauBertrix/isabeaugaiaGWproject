# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:27:30 2018

@author: isabeau
"""

import numpy as np
import random as rd
import pylab as pl
#import matplotlib.pyplot as plt
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
   
    mmin = 3.
    mmax = 4.
   
    cmin = -10.  # lower range on c (the same as the uniform c prior lower bound)
    cmax = 10.   # upper range on c (the same as the uniform c prior upper bound)

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

        m = mprime*(self.mmax-self.mmin) + self.mmin      # convert back to m
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
tol = 0.00000005 #stopping criteria, smaller longer but more accurate

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
area = len(mchain) * ( bin_edges[1] - bin_edges[0] )

cchain = solution.samples[:,1]
hist, bin_edges2 = np.histogram(cchain)
area2 = len(cchain) * ( bin_edges2[1] - bin_edges2[0] )
from scipy import stats
kernel = stats.gaussian_kde( mchain )
kernel2 = stats.gaussian_kde( cchain )
#print(mchain)
#print(bin_edges)
#print( kernel.integrate_box_1d(1,4) )


m_values = np.arange(3.490,3.508,0.000001)
c_values = np.arange(0.6,1.6, 0.001)
#print(area)
#print(len(mchain), len(cchain))
#plt.plot( m_values , kernel(m_values) * area )
#plt.hist(mchain, bins=bin_edges)
#plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/plot/test.png")
#plt.clf()

#plt.plot( c_values, kernel2(c_values) * area2)
#plt.hist(cchain, bins=bin_edges2)

#plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/plot/test2.png")
#plt.clf()
gridx = np.linspace(min(mchain),max(mchain),50)
gridy = np.linspace(min(cchain),max(cchain),50)

grid, _, _ = np.histogram2d(mchain, cchain, bins=[gridx, gridy])
left, width = 0.12, 0.55
bottom, height = 0.12, 0.55
bottom_h = left_h = left+width+0.02
 
# Set up the geometry of the three plots
rect_temperature = [left, bottom, width, height] # dimensions of temp plot
rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram
 
# Set up the size of the figure
#fig = plt.figure(1, figsize=(9.5,9))

#axTemperature = plt.axes(rect_temperature)
#axTemperature = plt.pcolormesh(gridx, gridy, grid, cmap = 'Greys')
#axTemperature = plt.axis([3.492, 3.506, 0.9, 1.7])


#axHistx = plt.axes(rect_histx)
#axHisty = plt.axes(rect_histy)
#axHistx.hist(mchain, color = 'blue')
#axHistx.plot( m_values , kernel(m_values) * area )
#axHistx.set_xlim((3.492, 3.506))


#axHisty.hist(cchain, orientation='horizontal', color = 'blue')
#axHisty.plot(kernel2(c_values) * area2, c_values)
#axHisty.set_ylim((0.9, 1.7))

#remove axis
#nullfmt = plt.NullFormatter()   
#axHistx.xaxis.set_major_formatter(nullfmt)
#axHistx.yaxis.set_major_formatter(nullfmt)
#axHisty.xaxis.set_major_formatter(nullfmt)
#axHisty.yaxis.set_major_formatter(nullfmt)
#plt.show()
#"plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/plot/swag.png")
#plt.clf()
