# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:04:07 2018

@author: isabeau
"""
working_directory = '/home/isabeau/Documents/Cours/isabeaugaiaGWproject/'

import numpy as np
from numpy import linalg as LA

import re
import os
import sys
import matplotlib.pyplot as plt
from collections import namedtuple

sys.path.append("functions/")

from Delta_n import *
from CoordinateConversion import *
from LoadData import *
from Delta_t import *
from calculate_timing_residual import *
from Add_Noise import *
from MATRIX import *
from save_result_to_file import *
 
day = 24 * 60 * 60.
year = 3660. * 24. * 365.25
week = 3660. * 24. * 7.
month = week * 4.
c = 2.99e8
w = np.pi/2

GW_parameters = namedtuple("GW_parameters", "logGWfrequency logAmplus logAmcross cosTheta Phi DeltaPhiPlus DeltaPhiCross")
GW_par = GW_parameters( logGWfrequency = np.log(2*np.pi/(3*month)), logAmplus = -12*np.log(10), logAmcross = -12*np.log(10), cosTheta = 0.5, Phi = 1.0, DeltaPhiPlus = 1*np.pi , DeltaPhiCross = np.pi )

star_positions_times_angles = LoadData( "MockAstrometricTimingData/gwastrometry-gaiasimu-1000-randomSphere-v2.dat" )
number_of_stars = len(star_positions_times_angles)
distances = np.random.normal(3.086e19 , 1.0e16, len(star_positions_times_angles))

changing_star_positions = []
for i in range(number_of_stars):
	changing_star_positions.append( [ delta_ncomplicated(star_positions_times_angles[i][0], t, GW_par, distances[i]) for t in star_positions_times_angles[i][1] ] )

timing_residuals = calculate_timing_residuals_simple( star_positions_times_angles, GW_par )    
sigma_t = 1.667 * 1.0e3 / np.sqrt(1.0e9/number_of_stars)

Sigma4 = fisher_matrix4(star_positions_times_angles , GW_par, sigma_t, distances)    
w4,v4 = LA.eigh( Sigma4 )
invSigma4 = np.dot( v4 , np.dot( np.diag(1./w4) , np.transpose(v4) )  )
error = np.sqrt(np.diag(invSigma4))
Save_Results_To_File ( invSigma4 , "invSigma4.dat" )
           
from pymultinest.solve import Solver

class GaiaModelPyMultiNest(Solver):

    # define the prior parameters
    logGWfrequencymin = -13
    logGWfrequencymax = -11
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

    def __init__(self, data, star_positions_times_angles, timing_residuals, sigma_t, distances, **kwargs):
        # set the data
	self._data = data
        self._star_positions_times_angles = star_positions_times_angles
        self._timing_residuals = timing_residuals
        self._number_of_stars = len(star_positions_times_angles)
        self._sigma_t = sigma_t      
        self._logsigma_t = np.log(sigma_t) 
        self._sigma_tsq = sigma_t * sigma_t
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
        
        logGWfrequency = logGWfrequencyprime*(self.logGWfrequencymax-self.logGWfrequencymin) + self.logGWfrequencymin
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
        
        logl = 0
        for i in range(self._number_of_stars): # loop over stars
            for j in range(len(self._star_positions_times_angles[i][1])): # loop over measurements
                measured_timing_residual_in_nanoseconds = self._timing_residuals[i][j]
                modelled_timing_residual_in_nanoseconds = 1.0e9 * calculate_delta_t_complicated( self._star_positions_times_angles[i][0] , 1.0e-9 * self._star_positions_times_angles[i][1][j] , self._star_positions_times_angles[i][2][j] , GW_par, distances[i] )
                x = measured_timing_residual_in_nanoseconds - modelled_timing_residual_in_nanoseconds
                logl = logl - (0.5 * x*x / self._sigma_tsq + LN2PI/2. + self._logsigma_t ) 
               
        return logl      
       
    

def TestLogLikelihood(star_positions_times_angles, timing_residuals, sigma_t, cube):
    GW_par = GW_parameters( logGWfrequency = cube[0], logAmplus = cube[1], logAmcross = cube[2], cosTheta = cube[3], Phi = cube[4], DeltaPhiPlus = cube[5] , DeltaPhiCross = cube[6] )
    number_of_stars = len(star_positions_times_angles)
    sigma_tsq = sigma_t * sigma_t
    logsigma_t = np.log(sigma_t)
    logl = 0
    for i in range(number_of_stars): # loop over stars
        for j in range(len(star_positions_times_angles[i][1])): # loop over measurements
            measured_timing_residual_in_nanoseconds = timing_residuals[i][j]
            modelled_timing_residual_in_nanoseconds = 1.0e9 * calculate_delta_t_complicated( star_positions_times_angles[i][0] , 1.0e-9 * star_positions_times_angles[i][1][j] , star_positions_times_angles[i][2][j] , GW_par , distances[i])
            x = measured_timing_residual_in_nanoseconds - modelled_timing_residual_in_nanoseconds
            logl = logl - (0.5 * x*x / sigma_tsq + LN2PI/2. + logsigma_t ) 
               
    return logl          
    
LN2PI = np.log(2.*np.pi)

timing_residuals = calculate_timing_residuals_complicated( star_positions_times_angles, GW_par, distances )

nlive = 512 #number of live points
ndim = 7 #number of parameters
tol = 0.5 #stopping criteria, smaller longer but more accurate

solution = GaiaModelPyMultiNest(changing_star_positions, star_positions_times_angles, timing_residuals, sigma_t, distances, n_dims=ndim, n_live_points=nlive, evidence_tolerance=tol, outputfiles_basename = '/home/isabeau/Documents/Cours/isabeaugaiaGWproject/delta_results/run4', verbose = True);
