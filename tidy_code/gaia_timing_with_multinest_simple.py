# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:09:39 2018
@author: isabeau
"""

import numpy as np
from numpy import linalg as LA

import re
import os
import sys
#import matplotlib.pyplot as plt
from mpi4py import MPI
from collections import namedtuple
sys.path.append("functions/")

from Delta_n import *
from CoordinateConversion import *
from LoadData import *
from Delta_t import *
from calculate_timing_residual import *
from Add_Noise import *
from derivatives import *
from MATRIX import *
from save_result_to_file import *

day = 24 * 60 * 60.
year = 3660. * 24. * 365.25
week = 3660. * 24. * 7.
month = week * 4.
c = 2.99e8
w = np.pi/2

GW_parameters = namedtuple ( "GW_parameters" , "logGWfrequency logAmplus logAmcross cosTheta Phi DeltaPhiPlus DeltaPhiCross" )
GW_par = GW_parameters ( logGWfrequency = np.log(2*np.pi/(3*month)) , logAmplus = -12*np.log(10) , logAmcross = -12*np.log(10) , cosTheta = 0.5 , Phi = 1.0 , DeltaPhiPlus = 1*np.pi , DeltaPhiCross = np.pi )

star_positions_times_angles = LoadData( "MockAstrometricTimingData/gwastrometry-gaiasimu-1000-randomSphere-v2.dat" )
number_of_stars = len ( star_positions_times_angles )


timing_residuals = calculate_timing_residuals_simple ( star_positions_times_angles, GW_par )
sigma_t = 1.667 * 1.0e3 / np.sqrt ( 1.0e9 / number_of_stars ) 

Sigma3 = fisher_matrix3 ( star_positions_times_angles , GW_par, sigma_t*1.0e-9 )
w3,v3 = LA.eigh ( Sigma3 )
invSigma3 = np.dot ( v3 , np.dot( np.diag(1./w3) , np.transpose(v3) )  )
error = np.sqrt ( np.diag ( invSigma3 ) )
Save_Results_To_File ( invSigma3 , "{}/invSigma3.dat".format(os.environ['outputfiles_dir']) )
   
from pymultinest.solve import Solver
from scipy.special import ndtri

LN2PI = np.log ( 2.0 * np.pi )

class GaiaModelPyMultiNest ( Solver ):

    # define the prior parameters
    logGWfrequencymin = np.log(2*np.pi/(3*month)) - 2.5 * error[0]
    logGWfrequencymax = np.log(2*np.pi/(3*month)) + 2.5 * error[0]
    logAmplusmin = -12*np.log(10) - 2.5 * error[1]
    logAmplusmax = -12*np.log(10) + 2.5 * error[1]
    logAmcrossmin = -12*np.log(10) - 2.5 * error[2]
    logAmcrossmax = -12*np.log(10) + 2.5 * error[2]
    cosThetamin = 0.5 - 2.5 * error[3]
    cosThetamax = 0.5 + 2.5 * error[3]
    Phimin = 1.0 - 2.5 * error[4]
    Phimax = 1.0 + 2.5 * error[4]
    DeltaPhiPlusmin = np.pi - 2.5 * error[5]
    DeltaPhiPlusmax = np.pi + 2.5 * error[5]
    DeltaPhiCrossmin = np.pi  - 2.5 * error[6]
    DeltaPhiCrossmax = np.pi  + 2.5 * error[6]

    def __init__(self, timing_residuals , star_positions_times_angles , sigma_t , **kwargs ):
        # set the data
	self._data = timing_residuals
        self._star_positions_times_angles = star_positions_times_angles
        self._number_of_stars = len ( star_positions_times_angles )
        self._sigma_t = sigma_t
        self._logsigma_t = np.log ( sigma_t )
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
        
        logGWfrequency = logGWfrequencyprime*(self.logGWfrequencymax-self.logGWfrequencymin) + self.logGWfrequencymin
        logAmplus = logAmplusprime*(self.logAmplusmax-self.logAmplusmin) + self.logAmplusmin 
        logAmcross = logAmcrossprime*(self.logAmcrossmax-self.logAmcrossmin) + self.logAmcrossmin 
        cosTheta = cosThetaprime*(self.cosThetamax-self.cosThetamin) + self.cosThetamin 
        Phi = Phiprime*(self.Phimax-self.Phimin) + self.Phimin 
        DeltaPhiPlus = DeltaPhiPlusprime*(self.DeltaPhiPlusmax-self.DeltaPhiPlusmin) + self.DeltaPhiPlusmin 
        DeltaPhiCross = DeltaPhiCrossprime*(self.DeltaPhiCrossmax-self.DeltaPhiCrossmin) + self.DeltaPhiCrossmin 
        
        return np.array ( [ logGWfrequency , logAmplus , logAmcross , cosTheta , Phi , DeltaPhiPlus , DeltaPhiCross ] )

    def LogLikelihood ( self , cube ):
        """
        The log likelihood function. This function has to be called "LogLikelihood".
        Args:
            cube (:class:`numpy.ndarray`): an array of parameter values.
        Returns:
            float: the log likelihood value.
        """
       
        GW_par_multinest = GW_parameters ( logGWfrequency = cube[0], logAmplus = cube[1], logAmcross = cube[2], cosTheta = cube[3], Phi = cube[4], DeltaPhiPlus = cube[5] , DeltaPhiCross = cube[6] )

        modelled_timing_residuals = calculate_timing_residuals_simple (self._star_positions_times_angles, GW_par_multinest )

        logl = 0
        for i in range ( self._number_of_stars ): # loop over stars
            for j in range ( len ( self._star_positions_times_angles[i][1] ) ): # loop over measurements
                measured_timing_residual_in_nanoseconds = self._data[i][j]
                modelled_timing_residual_in_nanoseconds = modelled_timing_residuals[i][j]
                x = measured_timing_residual_in_nanoseconds - modelled_timing_residual_in_nanoseconds
                logl = logl - ( 0.5 * x*x / self._sigma_tsq + LN2PI/2. + self._logsigma_t ) 
               
        return logl      
       
    

def TestLogLikelihood ( timing_residuals , star_positions_times_angles , sigma_t , cube ):

    GW_par_multinest = GW_parameters ( logGWfrequency = cube[0], logAmplus = cube[1], logAmcross = cube[2], cosTheta = cube[3], Phi = cube[4], DeltaPhiPlus = cube[5] , DeltaPhiCross = cube[6] )

    modelled_timing_residuals = calculate_timing_residuals_simple ( star_positions_times_angles, GW_par_multinest )
    
    number_of_stars = len ( star_positions_times_angles )
    sigma_tsq = sigma_t * sigma_t
    logsigma_t = np.log ( sigma_t )

    logl = 0
    for i in range ( number_of_stars ): # loop over stars
        for j in range ( len ( star_positions_times_angles[i][1] ) ): # loop over measurements
            measured_timing_residual_in_nanoseconds = timing_residuals[i][j]
            modelled_timing_residual_in_nanoseconds = modelled_timing_residuals[i][j]
            x = measured_timing_residual_in_nanoseconds - modelled_timing_residual_in_nanoseconds
            logl = logl - (0.5 * x*x / sigma_tsq + LN2PI/2. + logsigma_t ) 
               
    return logl


nlive = 512 #number of live points
ndim = 7 #number of parameters
tol = 0.5 #stopping criteria, smaller longer but more accurate

solution = GaiaModelPyMultiNest(timing_residuals,
                                star_positions_times_angles,
                                sigma_t,
                                n_dims=ndim,
                                n_live_points=nlive,
                                evidence_tolerance=tol,
                                outputfiles_basename="{}/1-".format(os.environ['outputfiles_dir']),
                                init_MPI=False,
                                verbose=True,
                                resume=False);
