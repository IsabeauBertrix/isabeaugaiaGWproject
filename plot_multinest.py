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
    values = np.linspace(minetmax[0],minetmax[1],1000)
    area = len(chain) * ( bin_edges[1] - bin_edges[0] )
    plt.hist(chain, bins = bin_edges)
    plt.plot( values , kernel(values) * area )
    plt.xlim(minetmax[0], minetmax[1])
    plt.show()
    return 1
    
def twoDhist( chain1 , chain2 , minetmax1, minetmax2):

    gridx = np.linspace(minetmax1[0],minetmax1[1],50)
    gridy = np.linspace(minetmax2[0],minetmax2[1],50)

    grid, _, _ = np.histogram2d(chain1, chain2, bins=[gridx, gridy])
    left, width = 0.12, 0.55
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width+0.02
    
    kernel1 = stats.gaussian_kde( chain1 )
    kernel2 = stats.gaussian_kde( chain2 )
    hist, bin_edges1 = np.histogram(chain1)
    hist, bin_edges2 = np.histogram(chain2)
    # Set up the geometry of the three plots
    rect_temperature = [left, bottom, width, height] # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram
 
    values1 = np.linspace(minetmax1[0],minetmax1[1],1000)
    area1 = len(chain1) * ( bin_edges1[1] - bin_edges1[0] )
    values2 = np.linspace(minetmax2[0],minetmax2[1],1000)
    area2 = len(chain2) * ( bin_edges2[1] - bin_edges2[0] )
    # Set up the size of the figure
    fig = plt.figure(1, figsize=(9.5,9))

    axTemperature = plt.axes(rect_temperature)
    axTemperature = plt.pcolormesh(gridx, gridy, grid, cmap = 'Greys')
    #axTemperature = plt.axis([minetmax1[0],minetmax1[1], minetmax2[0], minetmax2[1]])
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    axHistx.hist(chain1, color = 'blue')
    axHistx.plot( values1 , kernel1(values1) * area1 )
    axHistx.set_xlim((minetmax1[0], minetmax1[1]))


    axHisty.hist(chain2, orientation='horizontal', color = 'blue')
    axHisty.plot(kernel2(values2) * area2, values2)
    axHisty.set_ylim((minetmax2[0], minetmax2[1]))

#remove axis
    nullfmt = plt.NullFormatter()   
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHistx.yaxis.set_major_formatter(nullfmt)
    axHisty.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    plt.show()
    return 1


# load data from file
filename = "delta_results/run1post_equal_weights.dat"
multinest_data = np.loadtxt(filename)
npar = len(multinest_data[0])
chains = [multinest_data[:,i] for i in range(npar - 1)]
#filename2 = "out/minetmax.dat"
#minetmax = np.loadtxt(filename2)
minetmax = [[min(chains[i]), max(chains[i])] for i in range(npar - 1)]
# loop over params to produce 1D histograms
for i in range(npar - 1):  
    oneDhist(chains[i], minetmax[i])

# double loop over params to produce 2D histograms
"""
for i in range(2):
    for j in range(2):
        if j != i:
            twoDhist(chains[i], chains[j], minetmax[i], minetmax[j])
 """       