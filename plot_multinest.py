# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:03:11 2018

@author: isabeau
""" 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import namedtuple
import re
import os

GW_parameters = namedtuple("GW_parameters", "logGWfrequency logAmplus logAmcross cosTheta Phi DeltaPhiPlus DeltaPhiCross")

def oneDhist( chain, minetmax, sigma, mu ):
    x = np.linspace(minetmax[0],minetmax[1],7)
    
    y = 1000 * np.exp( -0.5 * (x-mu)*(x-mu)) #/ (sigma* sigma))#np.exp(-0.5 * (x - chain) / (sigma * sigma))
    hist, bin_edges = np.histogram(chain)
    
    #area = len(chain) * ( bin_edges[1] - bin_edges[0] )
    plt.hist(chain, bins = bin_edges)
    plt.plot( x , y )
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

def cartesian_coordinate_from_latitude_and_longitude(l,b):
    theta = np.pi/2. - b * np.pi / 180.
    phi = l * np.pi / 180.
    x =  np.sin(theta) * np.cos(phi)
    y =  np.sin(phi) * np.sin(theta)
    z =  np.cos(theta)
    return np.array([x,y,z])
# load data from file
filename = "chains-21947/1-post_equal_weights.dat"
multinest_data = np.loadtxt(filename)
npar = len(multinest_data[0])
chains = [multinest_data[:,i] for i in range(npar - 1)]

testfilename ="chains-21947/1-stats.dat"

def LoadData( filename ):

    if ( os.path.isfile( filename ) == False ):
        print "Error: file does not exist"
        return 0

    with open( filename ) as f:
        content = f.readlines()

    data = []

    for i in range( 100 ):#len( content ) ):

        line = content[i]
        line = re.split(', \[|\], \]|\]',line)

        SkyPosition = np.array( [ float( re.split(', ',line[0])[index] ) for index in [0,1] ] )
        
        SkyPosition = cartesian_coordinate_from_latitude_and_longitude(SkyPosition[0],SkyPosition[1])       
        
        Times = np.array( [ np.uint64(a) for a in re.split(', ',line[1]) ] )

        ScanAngles = np.array( [ float(a) for a in re.split(', ',line[3]) ] )

        if ( len(Times) != len(ScanAngles) ):
            print "Error: something bad has happened in LoadData()"
            return 0

        data.append( [ SkyPosition , Times , ScanAngles] )

    return data
    
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
print(Load_MultiNest_Stats_File(testfilename))

def delta_n ( n , t, GW_par ):
    epsilon_theta = np.array([GW_par.cosTheta*np.cos(GW_par.Phi), GW_par.cosTheta*np.sin(GW_par.Phi) , -np.sqrt(1-np.power(GW_par.cosTheta,2))])
    epsilon_phi = np.array([-np.sin(GW_par.Phi), np.cos(GW_par.Phi), 0])
    q = np.array([np.sqrt(1-np.power(GW_par.cosTheta,2)) * np.cos(GW_par.Phi), np.sqrt(1-np.power(GW_par.cosTheta,2)) * np.sin(GW_par.Phi),GW_par.cosTheta])
    epsilon_plus= np.outer(epsilon_theta, epsilon_theta) - np.outer(epsilon_phi, epsilon_phi)
    epsilon_cross= np.outer(epsilon_theta, epsilon_phi) + np.outer(epsilon_phi,epsilon_theta)

    H = (np.exp(GW_par.logAmplus)*np.exp(1j*GW_par.DeltaPhiPlus)*epsilon_plus + np.exp(GW_par.logAmcross)*np.exp(1j*GW_par.DeltaPhiCross)*epsilon_cross )*np.exp(1j*t*np.exp(GW_par.logGWfrequency))
    
    return np.real((n-q)/(2*(1-np.dot(q,n)))*np.dot(n,np.dot(H,n))-0.5*np.dot(H,n))
    
def calculate_delta_t_simple(n, t, psi, GW_par ):
 
    # spherical polar coordinates of the star
    phi = np.arctan2(n[1],n[0])
    theta = np.arccos(n[2])
    
    # basis vectors on the sky
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    e_theta = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    
    # direction of Gaia scan
    x = np.sin(psi)*e_phi - np.cos(psi)*e_theta
    
    # astrometric deflection
    dn = delta_n( n , t, GW_par )
    
    # gaia's angular rotation frequency (rad/s) period = 6 hours
    w = 2. * np.pi / ( 6. * 60. * 60. )
    
    # compute the time delay
    return np.dot(dn,x) / w
    #IT IS IN SECONDS
def calculate_timing_residuals_simple ( star_positions_times_angles , GW_par ):

    x = []
    for j in range( len( star_positions_times_angles ) ): # loop over stars
        n = star_positions_times_angles[j][0]
        x.append( [ 1.0e9 * calculate_delta_t_simple(n, 1.0e-9 * star_positions_times_angles[j][1][i], star_positions_times_angles[j][2][i], GW_par, distances) for i in range(len(star_positions_times_angles[j][1])) ] ) # loop of measurements of each star
    
    return np.array( x )

def derivative1( n , t , GW_par, param_index, scale ):
    deltas = [np.power( 10 , -10.5), np.power(10, -4.75) ,np.power(10, -4.75), np.power(10 , -5.25), np.power(10 , -5.25), np.power( 10 , -4.99), np.power(10 , -4.99)]
    
    if param_index == 0:
        GW = GW_par._asdict()
        GW['logGWfrequency'] = GW['logGWfrequency'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = delta_n( n , t , GW )
        GW = GW_par._asdict()
        GW['logGWfrequency'] = GW['logGWfrequency'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - delta_n( n, t, GW)
        return answer / (2 * deltas[param_index] * scale) 
    elif param_index == 1:
        GW = GW_par._asdict()
        GW['logAmplus'] = GW['logAmplus'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = delta_n( n , t , GW )
        GW = GW_par._asdict()
        GW['logAmplus'] = GW['logAmplus'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - delta_n( n, t, GW )
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 2:
        GW = GW_par._asdict()
        GW['logAmcross'] = GW['logAmcross'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = delta_n( n , t, GW )
        GW = GW_par._asdict()
        GW['logAmcross'] = GW['logAmcross'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - delta_n( n, t, GW)
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 3:
        GW = GW_par._asdict()
        GW['cosTheta'] = GW['cosTheta'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = delta_n( n , t , GW )
        GW = GW_par._asdict()
        GW['cosTheta'] = GW['cosTheta'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - delta_n( n, t, GW)
        return answer / (2 * deltas[param_index] * scale) 
    elif param_index == 4:
        GW = GW_par._asdict()
        GW['Phi'] = GW['Phi'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = delta_n( n , t , GW )
        GW = GW_par._asdict()
        GW['Phi'] = GW['Phi'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - delta_n( n, t, GW)
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 5:
        GW = GW_par._asdict()
        GW['DeltaPhiPlus'] = GW['DeltaPhiPlus'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = delta_n( n , t , GW )
        GW = GW_par._asdict()
        GW['DeltaPhiPlus'] = GW['DeltaPhiPlus'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - delta_n( n, t, GW)
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 6:
        GW = GW_par._asdict()
        GW['DeltaPhiCross'] = GW['DeltaPhiCross'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = delta_n( n , t , GW )
        GW = GW_par._asdict()
        GW['DeltaPhiCross'] = GW['DeltaPhiCross'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - delta_n( n, t, GW)
        return answer / (2 * deltas[param_index] * scale) 
    else:
        print('error')
        return(-1)

def matrix_derivative1(n , t , GW_par):
    v1 = [derivative1( n , t , GW_par, param_index, 1.0 ) for param_index in range(7)]
    u1 = np.zeros((7 , 7))
    for i in range ( 7 ):
        for j in range ( 7 ):
            u1[i][j] = np.dot( v1[i] , v1[j] )
            
    return u1

def fisher_matrix1 (star_positions_times_angles , GW_par, sigma):
    number_of_stars = len(star_positions_times_angles)
    Sigma1 = np.zeros(( 7 , 7 ))
    for i in range( number_of_stars):
        for j in range( len(star_positions_times_angles[i][2])): #len of the angles
            M = matrix_derivative1( star_positions_times_angles[i][0], star_positions_times_angles[i][1][j] * 1.0e-9 , GW_par)
            Sigma1 = Sigma1 + M / (sigma * sigma )
    return(Sigma1) 

star_positions_times_angles = LoadData( "MockAstrometricTimingData/gwastrometry-gaiasimu-1000-randomSphere-v2.dat" )
sigma = 2.9e-13 
sigma_t = 1.0e-9     
distances = np.random.normal(3.086e16 , 1.0e13, len(star_positions_times_angles))
mu = Load_MultiNest_Stats_File(testfilename)
print(mu)
"""
exit(-1)    
SIGMA = fisher_matrix1 (star_positions_times_angles , mu , sigma )
 
from numpy import linalg as LA
w1,v1 = LA.eigh( SIGMA )
invSigma1 = np.dot( v1 , np.dot( np.diag(1./w1) , np.transpose(v1) )  )
SIGMA1 = invSigma1
"""
minetmax = [[min(chains[i]), max(chains[i])] for i in range(npar - 1)]
# loop over params to produce 1D histograms
for i in range(7):  
    oneDhist(chains[i], minetmax[i], sigma, mu)

# double loop over params to produce 2D histograms
"""
for i in range(2):
    for j in range(2):
        if j != i:
            twoDhist(chains[i], chains[j], minetmax[i], minetmax[j])
 """       