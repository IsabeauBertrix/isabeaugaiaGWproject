
"""
Created on Fri Mar 16 17:04:07 2018
@author: isabeau
"""
import numpy as np
from numpy import linalg as LA

import random as rd
import pylab as pl
import matplotlib.pyplot as plt
import re
import os
import sys 
from collections import namedtuple

sys.path.append("functions/")

from CoordinateConversion import *
from LoadData import *
from Delta_n import *
from Add_Noise import *
from gen_rand_point import *


GW_parameters = namedtuple("GW_parameters", "logGWfrequency logAmplus logAmcross cosTheta Phi DeltaPhiPlus DeltaPhiCross")
GW_par = GW_parameters( logGWfrequency = np.log( 2 ) - 7 * np.log(10), logAmplus = np.log(3) - 14*np.log(10), logAmcross = np.log(3) - 14*np.log(10), cosTheta = 0.5, Phi = 1.0, DeltaPhiPlus = 1*np.pi , DeltaPhiCross = np.pi/2. )
    

star_positions_times_angles = LoadData( "MockAstrometricTimingData/gwastrometry-gaiasimu-1000-randomSphere-v2.dat" )
sigma = 2.9e-13   
distances = np.random.normal(3.086e16 , 1.0e13, len(star_positions_times_angles))


     
number_of_stars = 1000
star_positions = [gen_rand_point() for i in range(number_of_stars)]

day = 24 * 60 * 60.
year = 3660. * 24. * 365.25
week = 3660. * 24. * 7.
month = week * 4.
measurement_times = np.arange(0, 6*month, 1*week)

#GW_par = GW_parameters( logGWfrequency = np.log(2*np.pi/(day)), logAmplus = np.log(0.5), logAmcross = np.log(0.5), cosTheta = 1.0, Phi = 1.0, DeltaPhiPlus = 0 , DeltaPhiCross = 0 )
        
changing_star_positions = np.array([ [ delta_ncomplicated(star_positions[i], t, GW_par, distances[i]) for i in range(number_of_stars)] for t in measurement_times] )

microarcsecond = np.pi/(180*3600*1e6)
sigma = 100 * microarcsecond / np.sqrt(1.0e9/number_of_stars)
#changing_star_positions = changing_star_positions + noise(star_positions, measurement_times, sigma)


from pymultinest.solve import Solver
from scipy.special import ndtri  

LN2PI = np.log(2.*np.pi)

class GaiaModelPyMultiNest(Solver):
# define the prior parameters
   
    logGWfrequencymin = -8
    logGWfrequencymax = -6
    logAmplusmin = -14
    logAmplusmax = -13
    logAmcrossmin = -14
    logAmcrossmax = -13
    cosThetamin = -1
    cosThetamax = 1
    Phimin = 0
    Phimax = 2*np.pi
    DeltaPhiPlusmin= 0
    DeltaPhiPlusmax = 2*np.pi
    DeltaPhiCrossmin= 0
    DeltaPhiCrossmax = 2*np.pi


    def __init__(self, data, sky_positions, measurement_times, sigma, distances, **kwargs):
        # set the data
        self._data = data        
        self._sky_positions = sky_positions 
        self._number_of_stars = len(sky_positions)
        self._measurement_times = measurement_times
        self._sigma = sigma      
        self._logsigma = np.log(sigma) 
        self._sigmasq = sigma * sigma
	self._distances = distances
        
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


        # calculate the model
        model_sky_positions = np.array([ [ delta_ncomplicated(self._sky_positions[i], t, GW_par, self._distances[i]) for i in range(self._number_of_stars)] for t in self._measurement_times] )


        logl = 0
        for i in range(self._number_of_stars):
            for j in range(len(self._measurement_times)):
                x = model_sky_positions[j][i] - self._data[j][i] 
                logl = logl - (0.5 * np.dot(x,x)/self._sigmasq + LN2PI + 2 * self._logsigma ) 
          
        return logl
        
        
def TestLogLikelihood(data, sky_positions, measurement_times, sigma, cube, distances):
      
        sigmasq = sigma * sigma
        logsigma = np.log(sigma)
        GW_par = GW_parameters( logGWfrequency = cube[0], logAmplus = cube[1], logAmcross = cube[2], cosTheta = cube[3], Phi = cube[4], DeltaPhiPlus = cube[5] , DeltaPhiCross = cube[6] )


        # calculate the model
        model_sky_positions = np.array([ [ delta_ncomplicated(sky_positions[i], t, GW_par, distances[i]) for i in range(number_of_stars)] for t in measurement_times] )


        logl = 0
        for i in range(number_of_stars):
            for j in range(len(measurement_times)):
                x = model_sky_positions[j][i] - data[j][i] 
                logl = logl - (0.5 * np.dot(x,x)/sigmasq + LN2PI + 2 * logsigma ) 
          
        return logl   

nlive = 1024  #number of live points
ndim = 7 #number of parameters (n and c here)
tol = 0.5 #stopping criteria, smaller longer but more accurate



solution = GaiaModelPyMultiNest(changing_star_positions, star_positions, measurement_times, sigma, distances, n_dims=ndim,
                                        n_live_points=nlive, evidence_tolerance=tol, outputfiles_basename = '/home/isabeau/Documents/Cours/isabeaugaiaGWproject/delta_results/run1');
