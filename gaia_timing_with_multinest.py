# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:04:07 2018

@author: isabeau
"""

import numpy as np
import re
import os
import matplotlib.pyplot as plt
from collections import namedtuple
    
def delta_n ( n , t, GW_par ):
 
    # basis vectors
    epsilon_theta = np.array([GW_par.cosTheta*np.cos(GW_par.Phi), GW_par.cosTheta*np.sin(GW_par.Phi) , -np.sqrt(1-np.power(GW_par.cosTheta,2))])
    epsilon_phi = np.array([-np.sin(GW_par.Phi), np.cos(GW_par.Phi), 0])

    # direction to GW source
    q = np.array([np.sqrt(1-np.power(GW_par.cosTheta,2)) * np.cos(GW_par.Phi), np.sqrt(1-np.power(GW_par.cosTheta,2)) * np.sin(GW_par.Phi),GW_par.cosTheta])
    
    # basis tensors
    epsilon_plus= np.outer(epsilon_theta, epsilon_theta) - np.outer(epsilon_phi, epsilon_phi)
    epsilon_cross= np.outer(epsilon_theta, epsilon_phi) + np.outer(epsilon_phi,epsilon_theta)

    # metric perturbation
    H = (np.exp(GW_par.logAmplus)*np.exp(1j*GW_par.DeltaPhiPlus)*epsilon_plus + np.exp(GW_par.logAmcross)*np.exp(1j*GW_par.DeltaPhiCross)*epsilon_cross )*np.exp(1j*t*np.exp(GW_par.logGWfrequency))
    
    # compute astrometric deflection, delta_n
    return np.real((n-q)/(2*(1-np.dot(q,n)))*np.dot(n,np.dot(H,n))-0.5*np.dot(H,n))

def orthographic_projection_north(p):
    if p[2]>0:
        return [p[0], p[1]]
    else:
        return [None, None]

# input: l and b are galactic longitude and latitude in degrees 
# output: unit cartesian vectors x y z 
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

    for i in range( 10 ): #len(content)

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
    
def calculate_delta_t(n, t, psi, GW_par):
 
    # spherical polar coordinates of the star
    phi = np.arctan2(n[1],n[0])
    theta = np.arccos(n[2])
    
    # basis vectors on the sky
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    e_theta = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    
    # direction of Gaia scan
    x = np.sin(psi)*e_phi - np.cos(psi)*e_theta
    
    # astrometric deflection
    dn = delta_n ( n , t, GW_par )
    
    # gaia's angular rotation frequency (rad/s) period = 6 hours
    w = 2. * np.pi / ( 6. * 60. * 60. )
    
    # compute the time delay
    return np.dot(dn,x) / w

def calculate_timing_residuals ( star_positions_times_angles , GW_par ):

    x = []
    for j in range( len( star_positions_times_angles ) ): # loop over stars
        n = star_positions_times_angles[j][0]
        x.append( [ 1.0e9 * calculate_delta_t(n, 1.0e-9 * star_positions_times_angles[j][1][i], star_positions_times_angles[j][2][i], GW_par) for i in range(len(star_positions_times_angles[j][1])) ] ) # loop of measurements of each star
    
    return np.array( x )
    
def inject_fake_noise( timing_residuals , sigma_t ):
   
    for j in range( len( timing_residuals ) ): # loop over stars
        for i in range( len( timing_residuals[j] ) ): # loop of measurements of each star
            
            delta_t = np.random.normal(0, sigma_t) # sigma_t is in nanoseconds
            timing_residuals[j][i] = timing_residuals[j][i] + delta_t
    return timing_residuals
   
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

    def __init__(self, star_positions_times_angles, timing_residuals, sigma_t, **kwargs):
        # set the data
        self._star_positions_times_angles = star_positions_times_angles
        self._timing_residuals = timing_residuals
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
                modelled_timing_residual_in_nanoseconds = 1.0e9 * calculate_delta_t( self._star_positions_times_angles[i][0] , 1.0e-9 * self._star_positions_times_angles[i][1][j] , self._star_positions_times_angles[i][2][j] , GW_par )
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
            modelled_timing_residual_in_nanoseconds = 1.0e9 * calculate_delta_t( star_positions_times_angles[i][0] , 1.0e-9 * star_positions_times_angles[i][1][j] , star_positions_times_angles[i][2][j] , GW_par )
            x = measured_timing_residual_in_nanoseconds - modelled_timing_residual_in_nanoseconds
            logl = logl - (0.5 * x*x / sigma_tsq + LN2PI/2. + logsigma_t ) 
               
    return logl          
    
    

    
LN2PI = np.log(2.*np.pi)

hour = 60. * 60.
day = hour * 24.
week = day * 7.
month = week * 4.
year = day * 365.25

star_positions_times_angles = LoadData( "MockAstrometricTimingData/gwastrometry-gaiasimu-1000-randomSphere-v2.dat" )

GW_parameters = namedtuple("GW_parameters", "logGWfrequency logAmplus logAmcross cosTheta Phi DeltaPhiPlus DeltaPhiCross")

GW_par = GW_parameters( logGWfrequency = np.log(2*np.pi/(3*month)), logAmplus = -12*np.log(10), logAmcross = -12*np.log(10), cosTheta = 0.5, Phi = 1.0, DeltaPhiPlus = 1 * np.pi , DeltaPhiCross = 1 * np.pi )

timing_residuals = calculate_timing_residuals( star_positions_times_angles, GW_par )

sigma_t = 1.6 # nanoseconds
#timing_residuals = inject_fake_noise(timing_residuals, sigma_t)







y = []
x = []

step_size = 0.0001
for i in range( 1000 ):
    cube = np.array( [ GW_par.logGWfrequency + step_size*(i-500), GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross ] )
    x.append( GW_par.logGWfrequency + step_size*(i-500) )
    y.append( TestLogLikelihood(star_positions_times_angles, 
timing_residuals, sigma_t, cube) )

y=y-max(y) # this line shifts all the log-likelihood values by a constant so the maximum value is logl=0
plt.plot(x,np.exp(y)) # we want to plot the likelihood (not log-likelihood) so we need to use np.exp here
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/timing_frequency.png")
plt.clf()


y = []
Y = []
x = []
X = []
step_size = 0.0001
for i in range( 1000 ):
    cube = np.array( [ GW_par.logGWfrequency, GW_par.logAmplus + step_size*(i-500), GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross ] )
    x.append(GW_par.logAmplus + step_size*(i-500)) 
    X.append(GW_par.logAmcross + step_size*(i-500))
    y.append(TestLogLikelihood(star_positions_times_angles, timing_residuals, sigma_t, cube))
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross + step_size*(i-500), GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross])
    Y.append(TestLogLikelihood(star_positions_times_angles, timing_residuals, sigma_t, cube))
Y = Y - max(Y)
y = y - max(y)
plt.plot(x,np.exp(y))
plt.plot(X, np.exp(Y))
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/timing_amplitude.png")
plt.clf()

y = []
x = []

step_size = 0.0001
for i in range( 1000 ):
    cube = np.array( [ GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta + step_size*(i-500), GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross ] )
    x.append( GW_par.cosTheta + step_size*(i-500) )
    y.append( TestLogLikelihood(star_positions_times_angles, timing_residuals, sigma_t, cube) )

y=y-max(y) # this line shifts all the log-likelihood values by a constant so the maximum value is logl=0
plt.plot(x,np.exp(y)) # we want to plot the likelihood (not log-likelihood) so we need to use np.exp here
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/timing_cosTheta.png")
plt.clf()

y = []
x = []

step_size = 0.0001
for i in range( 1000 ):
    cube = np.array( [ GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi + step_size*(i-500), GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross ] )
    x.append( GW_par.Phi + step_size*(i-500) )
    y.append( TestLogLikelihood(star_positions_times_angles, timing_residuals, sigma_t, cube) )

y=y-max(y) # this line shifts all the log-likelihood values by a constant so the maximum value is logl=0
plt.plot(x,np.exp(y)) # we want to plot the likelihood (not log-likelihood) so we need to use np.exp here
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/timing_Phi.png")
plt.clf()

y = []
Y = []
x = []
X = []
step_size = 0.0001
for i in range( 1000 ):
    cube = np.array( [ GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus + step_size*(i-500), GW_par.DeltaPhiCross ] )
    x.append(GW_par.DeltaPhiPlus + step_size*(i-500)) 
    X.append(GW_par.DeltaPhiCross + step_size*(i-500))
    y.append(TestLogLikelihood(star_positions_times_angles, timing_residuals, sigma_t, cube))
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross + step_size*(i-500)])
    Y.append(TestLogLikelihood(star_positions_times_angles, timing_residuals, sigma_t, cube))
Y = Y - max(Y)
y = y - max(y)
plt.plot(x,np.exp(y))
plt.plot(X, np.exp(Y))
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/timing_deltaphi.png")
plt.clf()

exit(-1)









nlive = 1024 #number of live points
ndim = 7 #number of parameters
tol = 0.5 #stopping criteria, smaller longer but more accurate

solution = GaiaModelPyMultiNest(star_positions_times_angles, timing_residuals, sigma_t, n_dims=ndim, n_live_points=nlive, evidence_tolerance=tol, outputfiles_basename = '/home/isabeau/Documents/Cours/isabeaugaiaGWproject/delta_results/run1', verbose = True);
