# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:53:37 2018

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
from derivatives import *
from MATRIX import *
from save_result_to_file import *

outdir = '/home/isabeau/Documents/Cours/isabeaugaiaGWproject/delta_results/run1'

day = 24 * 60 * 60.
year = 3660. * 24. * 365.25
week = 3660. * 24. * 7.
month = week * 4.

GW_parameters = namedtuple("GW_parameters", "logGWfrequency logAmplus logAmcross cosTheta Phi DeltaPhiPlus DeltaPhiCross")
GW_par = GW_parameters( logGWfrequency = np.log(2*np.pi/(3*month)), logAmplus = -12*np.log(10), logAmcross = -12*np.log(10), cosTheta = 0.5, Phi = 1.0, DeltaPhiPlus = 1*np.pi , DeltaPhiCross = np.pi )

star_positions_times_angles = LoadData( "MockAstrometricTimingData/gwastrometry-gaiasimu-1000-randomSphere-v2.dat" )
    
number_of_stars = len(star_positions_times_angles)

changing_star_positions = []
for i in range(number_of_stars):
	changing_star_positions.append( [ delta_n(star_positions_times_angles[i][0], t, GW_par) for t in star_positions_times_angles[i][1] ] )

microarcsecond = np.pi/(180*3600*1e6)
sigma = 100 * microarcsecond / np.sqrt(1.0e9/number_of_stars)
#changing_star_positions = changing_star_positions + noise(star_positions, measurement_times, sigma)

Sigma1 = fisher_matrix1 (star_positions_times_angles , GW_par, sigma)    
w1,v1 = LA.eigh( Sigma1 )
invSigma1 = np.dot( v1 , np.dot( np.diag(1./w1) , np.transpose(v1) )  )
error = np.sqrt(np.diag(invSigma1))

Save_Results_To_File ( invSigma1 , "invSigma1.dat" )

from pymultinest.solve import Solver
from scipy.special import ndtri  

LN2PI = np.log(2.*np.pi)

class GaiaModelPyMultiNest(Solver):
# define the prior parameters
   
    logGWfrequencymin = np.log(2*np.pi/(3*month)) - 3 * error[0]
    logGWfrequencymax = np.log(2*np.pi/(3*month)) + 3 * error[0]
    logAmplusmin = -12*np.log(10) - 3 * error[1]
    logAmplusmax = -12*np.log(10) + 3 * error[1]
    logAmcrossmin = -13*np.log(10) - 1.0e-6
    logAmcrossmax = -13*np.log(10) + 1.0e-6
    cosThetamin = 0.5 - 1.0e-6
    cosThetamax = 0.5 + 1.0e-6
    Phimin = 1.0 - 1.0e-6
    Phimax = 1.0 + 1.0e-6
    DeltaPhiPlusmin = np.pi - 1.0e-6
    DeltaPhiPlusmax = np.pi + 1.0e-6
    DeltaPhiCrossmin = np.pi  - 1.0e-6
    DeltaPhiCrossmax = np.pi + 1.0e-6


    def __init__(self, data, star_positions_times_angles, sigma, **kwargs):
        # set the data
        self._data = data        
        self._star_positions_times_angles = star_positions_times_angles 
        self._number_of_stars = len(star_positions_times_angles)
        self._sigma = sigma      
        self._logsigma = np.log(sigma) 
        self._sigmasq = sigma * sigma
        
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
        model_sky_positions = []
	for i in range(self._number_of_stars):
		model_sky_positions.append( [ delta_n(self._star_positions_times_angles[i][0], t, GW_par) for t in self._star_positions_times_angles[i][1] ] )

        logl = 0
        for i in range(self._number_of_stars):
            for j in range(len(self._data[i])):
                x = model_sky_positions[i][j] - self._data[i][j] 
                logl = logl - (0.5 * np.dot(x,x)/self._sigmasq + LN2PI + 2 * self._logsigma )  
        return logl
        
        
def TestLogLikelihood(data, sky_positions, measurement_times, sigma, cube):
      
        sigmasq = sigma * sigma
        logsigma = np.log(sigma)
        GW_par = GW_parameters( logGWfrequency = cube[0], logAmplus = cube[1], logAmcross = cube[2], cosTheta = cube[3], Phi = cube[4], DeltaPhiPlus = cube[5] , DeltaPhiCross = cube[6] )


        # calculate the model
        model_sky_positions = np.array([ [ delta_n(sky_positions[i], t, GW_par) for i in range(number_of_stars)] for t in measurement_times] )


        logl = 0
        for i in range(number_of_stars):
            for j in range(len(measurement_times)):
                x = model_sky_positions[j][i] - data[j][i] 
                logl = logl - (0.5 * np.dot(x,x)/sigmasq + LN2PI + 2 * logsigma ) 
          
        return logl   

nlive = 512 #number of live points
ndim = 7 #number of parameters (n and c here)
tol = 0.5 #stopping criteria, smaller longer but more accurate

solution = GaiaModelPyMultiNest(changing_star_positions, star_positions_times_angles, sigma, n_dims=ndim,
                                        n_live_points=nlive, evidence_tolerance=tol, outputfiles_basename = outdir);
