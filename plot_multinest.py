# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:03:11 2018

@author: isabeau
""" 
import numpy as np
import random as rd
import pylab as pl
import matplotlib.pyplot as plt
from time import time
from scipy import stats
from pymultinest.solve import Solver
from scipy.special import ndtri 

def oneDhist( chain, minetmax ):
    kernel = stats.gaussian_kde( chain )
    hist, bin_edges = np.histogram(chain)
    values = np.arange(minetmax[0],minetmax[1],(minetmax[1]-minetmax[0])/1000.)
    area = len(chain) * ( bin_edges[1] - bin_edges[0] )
    plt.hist(chain, bins = bin_edges)
    plt.plot( values , kernel(values) * area )
    plt.xlim(minetmax[0], minetmax[1])
    plt.show()
    return 1
    
def twoDhist( chain1 , chain2 ):
    plt.hist(chain1, chain2)
    plt.show()
    return 1


# load data from file
filename = "out/post_equal_weights.dat"
multinest_data = np.loadtxt(filename)
npar = len(multinest_data[0])
chains = [multinest_data[:,i] for i in range(npar - 1)]
filename2 = "out/minetmax.dat"
minetmax = np.loadtxt(filename2)

# loop over params to produce 1D histograms
for i in range(npar - 1): 
    oneDhist(chains[i], minetmax[i])

# double loop over params to produce 2D histograms
for i in range(npar):
    for j in range(npar):
        twoDhist(chains[i], chains[j])
        
        
        
        
        
        
        
        
        
        
        
