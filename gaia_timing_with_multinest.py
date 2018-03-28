# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:04:07 2018

@author: isabeau
"""

import numpy as np
import re
import random as rd
import pylab as pl
import os
#import matplotlib.pyplot as plt

from time import time
from collections import namedtuple
 
   
def gen_rand_point(): 
    pi = np.random.normal(0, 1, 3)
    modulus_sqr_pi = np.dot(pi,pi)
    return pi/np.sqrt(modulus_sqr_pi)
    
    
def delta_n ( n , t, GW_par ):
    epsilon_theta = np.array([GW_par.cosTheta*np.cos(GW_par.Phi), GW_par.cosTheta*np.sin(GW_par.Phi) , -np.sqrt(1-np.power(GW_par.cosTheta,2))])
    epsilon_phi = np.array([-np.sin(GW_par.Phi), np.cos(GW_par.Phi), 0])
    q = np.array([np.sqrt(1-np.power(GW_par.cosTheta,2)) * np.cos(GW_par.Phi), np.sqrt(1-np.power(GW_par.cosTheta,2)) * np.sin(GW_par.Phi),GW_par.cosTheta])
    epsilon_plus= np.outer(epsilon_theta, epsilon_theta) - np.outer(epsilon_phi, epsilon_phi)
    epsilon_cross= np.outer(epsilon_theta, epsilon_phi) + np.outer(epsilon_phi,epsilon_theta)

    H = (np.exp(GW_par.logAmplus)*np.exp(1j*GW_par.DeltaPhiPlus)*epsilon_plus + np.exp(GW_par.logAmcross)*np.exp(1j*GW_par.DeltaPhiCross)*epsilon_cross )*np.exp(1j*t*np.exp(GW_par.logGWfrequency))
    
    return np.real((n-q)/(2*(1-np.dot(q,n)))*np.dot(n,np.dot(H,n))-0.5*np.dot(H,n))


def noise_single_star(n, t, sigma):
    x = np.array([rd.normalvariate(0,sigma), rd.normalvariate(0,sigma), rd.normalvariate(0,sigma)])
    x = x - np.dot(x, n) 
    return x


def noise(star_positions, measurement_times, sigma):
        return np.array([[noise_single_star(star_positions[i], t, sigma) for i in range(len(star_positions))] for t in measurement_times])


def orthographic_projection_north(p):
    if p[2]>0:
        return [p[0], p[1]]
    else:
        return [None, None]

#imput: l and b are galactic longitude and latitude in degrees 
#output: unit cartesian vectors x y z 
def cartesian_coordinate_from_latitude_and_longitude(l,b):
    theta = np.pi/2. - b * np.pi / 180.
    phi = l * np.pi / 180.
    x =  np.sin(theta) * np.cos(phi)
    y =  np.sin(phi) * np.sin(theta)
    z =  np.cos(theta)
    return np.array([x,y,z])

def LoadData( filename ):

    if ( os.path.isfile( filename ) == False ):
        print "Error: file does not exist"
        return 0

    with open( filename ) as f:
        content = f.readlines()

    data = []

    for i in range( 10 ):  #len(content)

        line = content[i]
        line = re.split(', \[|\], \]|\]',line)

        SkyPosition = np.array( [ float( re.split(', ',line[0])[index] ) for index in [0,1] ] )
        
        SkyPosition = cartesian_coordinate_from_latitude_and_longitude(SkyPosition[0],SkyPosition[1])       
        
        Times = (np.array( [ np.uint64(a) for a in re.split(', ',line[1]) ] ) - 63100000000000000) * 100

        ScanAngles = np.array( [ float(a) for a in re.split(', ',line[3]) ] )

        if ( len(Times) != len(ScanAngles) ):
            print "Error: something bad has happened in LoadData()"
            return 0

        data.append( [ SkyPosition , Times , ScanAngles] )

    return data
    
def calculate_delta_t(n,t,psi,GW_par):
    phi = np.arctan2(n[1],n[0])
    theta = np.arccos(n[2])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    e_theta = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    x = np.sin(psi)*e_phi - np.cos(psi)*e_theta
    dn = delta_n ( n , t, GW_par )
    w = 2*np.pi/(6*hour)
    delta_t = np.dot( dn,x) / w
    return(delta_t)

def inject_fake_gwsignal(star_positions_times_angles, GW_par):
    x = star_positions_times_angles
    for j in range(len(star_positions_times_angles)):
        line = x[j]
        for i in range(len(line[1])):
            delta_t = calculate_delta_t(line[0], line[1][i] * 1.0e-11 , line[2][i], GW_par ) * 1.0e11
            plusORminus = np.sign( delta_t)
            delta_t = np.uint64( abs( delta_t))
            if plusORminus < 0:
                x[j][1][i] = x[j][1][i] - delta_t
            elif plusORminus > 0:
                x[j][1][i] = x[j][1][i] + delta_t
            else:
                print("BAD THINGS HAVE HAPPENED")
    return x
    
def inject_fake_noise(star_positions_times_angles, sigma_t):
    x = star_positions_times_angles
    for j in range(len(star_positions_times_angles)):
        line = x[j]
        for i in range(len(line[1])):
            delta_t = np.random.normal(0, sigma_t) * 1.0e11
            plusORminus = np.sign( delta_t )
            delta_t = np.uint64( abs( delta_t ) ) 
            if plusORminus < 0:
                x[j][1][i] = x[j][1][i] - delta_t
            elif plusORminus > 0:
                x[j][1][i] = x[j][1][i] + delta_t
            else:
                print("BAD THINGS HAVE HAPPENED")
    return x
""" 
def plot_data(changing_star_positions):
    for i in range (number_of_stars): 
        p = star_positions[i] 
        delta_p = [changing_star_positions[j][i] for j in range(len(measurement_times))] 
        new_p = [orthographic_projection_north(p+delta_p[j]) for j in range(len(measurement_times))] 
        plt.plot( [ new_p[j][0] for j in range(len(measurement_times))] , [new_p[j][1] for j in range(len(measurement_times))] , 'r-' )
        plt.plot( [ new_p[j][0] for j in range(len(measurement_times))] , [new_p[j][1] for j in range(len(measurement_times))] , 'ro' )

    theta = pl.linspace(0, 2*np.pi, 40)
    x = np.cos(theta)
    y = np.sin(theta)
    plt.plot(x, y, 'k-')
    pl.axis("equal")
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    #plt.savefig("/home/isabeau/Documents/Cours/isabeauGaiaGWproject/first.png")
    #plt.clf()
    plt.show()
    
"""    





hour = 60 * 60.
day = 24 * 60 * 60.
year = 3660. * 24. * 365.25
week = 3660. * 24. * 7.
month = week * 4.


star_positions_times_angles = LoadData( "MockAstrometricTimingData/gwastrometry-gaiasimu-1000-randomSphere-v2.dat" )


GW_parameters = namedtuple("GW_parameters", "logGWfrequency logAmplus logAmcross cosTheta Phi DeltaPhiPlus DeltaPhiCross")
GW_par = GW_parameters( logGWfrequency = np.log(2*np.pi/(3*month)), logAmplus = -12*np.log(10), logAmcross = -13*np.log(10), cosTheta = 0.5, Phi = 1.0, DeltaPhiPlus = 1*np.pi , DeltaPhiCross = np.pi/2. )
        
sigma_t = 1.6e-9 #seconds

#print(calculate_delta_t([0,0,1], 10, 45, GW_par))
        
star_positions_times_angles = inject_fake_gwsignal(star_positions_times_angles, GW_par)
star_positions_times_angles = inject_fake_noise(star_positions_times_angles, sigma_t)


#plot_data(changing_star_positions)
#exit(-1)

from pymultinest.solve import Solver
from scipy.special import ndtri  

LN2PI = np.log(2.*np.pi)

class GaiaModelPyMultiNest(Solver):
# define the prior parameters
   
    logGWfrequencymin = -8
    logGWfrequencymax = -5
    logAmplusmin = -12*np.log(10) - 1.0e-6
    logAmplusmax = -12*np.log(10) + 1.0e-6
    logAmcrossmin = -13*np.log(10) - 1.0e-6
    logAmcrossmax = -13*np.log(10) + 1.0e-6
    cosThetamin = 0.5 - 1.0e-6
    cosThetamax = 0.5 + 1.0e-6
    Phimin = 1.0 - 1.0e-6
    Phimax = 1.0 + 1.0e-6
    DeltaPhiPlusmin = np.pi - 1.0e-6
    DeltaPhiPlusmax = np.pi + 1.0e-6
    DeltaPhiCrossmin = np.pi / 2 - 1.0e-6
    DeltaPhiCrossmax = np.pi / 2 + 1.0e-6


    def __init__(self, star_positions_times_angles, sigma_t, **kwargs):
        # set the data
        self._star_positions_times_angles = star_positions_times_angles 
        self._number_of_stars = len(star_positions_times_angles)
        self._sigma_t = sigma_t      
        self._logsigma_t = np.log(sigma_t) 
        self._sigma_tsq = sigma_t * sigma_t
        
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
        logGWfrequencyprime = cube[0]
        logAmplusprime = cube[1]
        logAmcrossprime = cube[2]
        cosThetaprime = cube[3]
        Phiprime = cube[4]
        DeltaPhiPlusprime = cube[5]
        DeltaPhiCrossprime = cube[6]
        
        logGWfrequency = logGWfrequencyprime*(self.logGWfrequencymax-self.logGWfrequencymin) + self.logGWfrequencymin      # convert back to m
        logAmplus = logAmplusprime*(self.logAmplusmax-self.logAmplusmin) + self.logAmplusmin 
        logAmcross = logAmcrossprime*(self.logAmcrossmax-self.logAmcrossmin) + self.logAmcrossmin 
        cosTheta = cosThetaprime*(self.cosThetamax-self.cosThetamin) + self.cosThetamin 
        Phi = Phiprime*(self.Phimax-self.Phimin) + self.Phimin 
        DeltaPhiPlus = DeltaPhiPlusprime*(self.DeltaPhiPlusmax-self.DeltaPhiPlusmin) + self.DeltaPhiPlusmin 
        DeltaPhiCross = DeltaPhiCrossprime*(self.DeltaPhiCrossmax-self.DeltaPhiCrossmin) + self.DeltaPhiCrossmin 
        
        return np.array([logGWfrequency, logAmplus, logAmcross, cosTheta, Phi, DeltaPhiPlus, DeltaPhiCross])

    def LogLikelihood(self, cube):
        """
        The log likelihood function. This function has to be called "LogLikelihood".

        Args:
            cube (:class:`numpy.ndarray`): an array of parameter values.

        Returns:
            float: the log likelihood value.
        """

      
       
        GW_par = GW_parameters( logGWfrequency = cube[0], logAmplus = cube[1], logAmcross = cube[2], cosTheta = cube[3], Phi = cube[4], DeltaPhiPlus = cube[5] , DeltaPhiCross = cube[6] )
        logl = - 0.5 * (GW_par.logGWfrequency - 7)*(GW_par.logGWfrequency - 7) / 0.01
        return logl
"""
        logl = 0
        for i in range(self._number_of_stars):
            for j in range(len(self._star_positions_times_angles[i][1])):
                x = self._star_positions_times_angles[i][1][j] - calculate_delta_t(self._star_positions_times_angles[i][0],self._star_positions_times_angles[i][1][j],self._star_positions_times_angles[i][2][j],GW_par)
                logl = logl - (0.5 * x*x / self._sigma_tsq + LN2PI/2. + self._logsigma_t ) 
"""       
       
    
nlive = 10 #1024 #number of live points
ndim = 7 #number of parameters (n and c here)
tol = 0.5 #stopping criteria, smaller longer but more accurate


solution = GaiaModelPyMultiNest(star_positions_times_angles, sigma_t, n_dims=ndim,
                                        n_live_points=nlive, evidence_tolerance=tol, outputfiles_basename = '/home/isabeau/Documents/Cours/isabeaugaiaGWproject/delta_results/run1');
