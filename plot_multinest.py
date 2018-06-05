# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:03:11 2018

@author: isabeau
""" 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import namedtuple


GW_parameters = namedtuple("GW_parameters", "logGWfrequency logAmplus logAmcross cosTheta Phi DeltaPhiPlus DeltaPhiCross")

def oneDhist( chain, minetmax, sigma, mu ):
    x = np.linspace(minetmax[0],minetmax[1],100)
    hist, bin_edges = np.histogram(chain, bins = 25)
 
    y = [len(chain) * (bin_edges[1] - bin_edges[0]) * np.exp( -0.5 * ( X - mu ) * ( X - mu ) / ( sigma * sigma ))/ np.sqrt(2 * np.pi * sigma * sigma) for X in x ] #np.exp(-0.5 * (x - chain) / (sigma * sigma))
    fig = plt.figure(1, figsize=(7,5))    
    
    plt.xlabel('log(Amplitude +)')
    plt.ylabel('Counts')
    
    
    #area = len(chain) * ( bin_edges[1] - bin_edges[0] )
    plt.hist(chain, bins = bin_edges, alpha = 0.5, color = 'blueviolet')
    plt.plot( x , y , linewidth = 2 , color = 'k')
    plt.xlim(minetmax[0], minetmax[1])
    #plt.show()
    
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

def Load_MultiNest_Stats_File ( filename ):

    with open(filename) as f:
        content = f.readlines() 
    
    content = content[4:11]
    
    content = [ c.split(' ') for c in content ]
    
    for i in range ( len ( content ) ):
        a = content[i]
        
        b = [ x for x in a if x !='' ]
        
        c = b[1]
        
        content[i] = float(c)
    
    return content  

def LoadFisherMatrix( Sigma ) : 
        
    return np.loadtxt(Sigma)
    
    
invSigma = LoadFisherMatrix('tidy_code/multinest_results/512-chains3-22307/invSigma3.dat')
sigma = np.sqrt(np.diag(invSigma))
    
# load data from file
filename = "tidy_code/multinest_results/512-chains3-22307/1-post_equal_weights.dat"
multinest_data = np.loadtxt(filename)
npar = len(multinest_data[0])
chains = [multinest_data[:,i] for i in range(npar - 1)]

testfilename ="tidy_code/multinest_results/512-chains3-22307/1-stats.dat"

mu = Load_MultiNest_Stats_File(testfilename)


minetmax = [[min(chains[i]), max(chains[i])] for i in range(npar - 1)]
# loop over params to produce 1D histograms
#for i in range(1):  
oneDhist(chains[0], minetmax[0], sigma[0], mu[0])
plt.savefig("timing_simple_amplitude.png")
plt.clf()
# double loop over params to produce 2D histograms
"""
for i in range(2):
    for j in range(2):
        if j != i:
            twoDhist(chains[i], chains[j], minetmax[i], minetmax[j])
 """      
