
"""
Created on Fri Mar 16 17:04:07 2018
@author: isabeau
"""

import numpy as np
import random as rd
import pylab as pl
import matplotlib.pyplot as plt

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
      
number_of_stars = 1000
star_positions = [gen_rand_point() for i in range(number_of_stars)]

day = 24 * 60 * 60.
year = 3660. * 24. * 365.25
week = 3660. * 24. * 7.
month = week * 4.
measurement_times = np.arange(0, 6*month, 1*week)

GW_parameters = namedtuple("GW_parameters", "logGWfrequency logAmplus logAmcross cosTheta Phi DeltaPhiPlus DeltaPhiCross")
GW_par = GW_parameters( logGWfrequency = np.log( 2 ) - 7 * np.log(10), logAmplus = np.log(3) - 14*np.log(10), logAmcross = np.log(3) - 14*np.log(10), cosTheta = 0.5, Phi = 1.0, DeltaPhiPlus = 1*np.pi , DeltaPhiCross = np.pi/2. )
#GW_par = GW_parameters( logGWfrequency = np.log(2*np.pi/(day)), logAmplus = np.log(0.5), logAmcross = np.log(0.5), cosTheta = 1.0, Phi = 1.0, DeltaPhiPlus = 0 , DeltaPhiCross = 0 )
        
changing_star_positions = np.array([ [ delta_n(star_positions[i], t, GW_par) for i in range(number_of_stars)] for t in measurement_times] )

microarcsecond = np.pi/(180*3600*1e6)
sigma = 100 * microarcsecond / np.sqrt(1.0e9/number_of_stars)
#changing_star_positions = changing_star_positions + noise(star_positions, measurement_times, sigma)

#plot_data(changing_star_positions)
#exit(-1)

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


    def __init__(self, data, sky_positions, measurement_times, sigma, **kwargs):
        # set the data
        self._data = data        
        self._sky_positions = sky_positions 
        self._number_of_stars = len(sky_positions)
        self._measurement_times = measurement_times
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
        model_sky_positions = np.array([ [ delta_n(self._sky_positions[i], t, GW_par) for i in range(self._number_of_stars)] for t in self._measurement_times] )


        logl = 0
        for i in range(self._number_of_stars):
            for j in range(len(self._measurement_times)):
                x = model_sky_positions[j][i] - self._data[j][i] 
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

nlive = 10 #1024 #number of live points
ndim = 7 #number of parameters (n and c here)
tol = 0.5 #stopping criteria, smaller longer but more accurate

"""
y = []
x = []
for i in range(100):
    cube = np.array([GW_par.logGWfrequency + 0.05*(i-50), GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross])
    x.append(GW_par.logGWfrequency + 0.05*(i-50)) 
    y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))

y = y - max(y)
plt.plot(x,np.exp(y))
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/gwfrequency.png")
plt.clf()
"""
y = []
Y = []
x = []
for i in range(100):
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus + 0.05*(i-50), GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross])
    x.append(GW_par.logAmplus + 0.05*(i-50)) 
    x.append(GW_par.logAmcross + 0.05*(i-50))
    y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross + 0.05*(i-50), GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross])
    Y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))
Y = Y - max(Y)
y = y - max(y)
plt.plot(x,np.exp(y))
plt.plot(x, np.exp(Y))
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/amplitude.png")
plt.clf()
"""
y = []
x = []
for i in range(100):
    cube = np.array([GW_par.logGWfrequency , GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta + 0.05*(i-50), GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross])
    x.append(GW_par.cosTheta + 0.05*(i-50)) 
    y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))

y = y - max(y)
plt.plot(x,np.exp(y))
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/costheta.png")
plt.clf()

y = []
x = []
for i in range(100):
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi + 0.05*(i-50), GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross])
    x.append(GW_par.Phi + 0.05*(i-50)) 
    y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))

y = y - max(y)
plt.plot(x,np.exp(y))
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/phi.png")
plt.clf()


y = []
Y = []
x = []
for i in range(100):
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus + 0.05*(i-50), GW_par.DeltaPhiCross])
    x.append(GW_par.DeltaPhiPlus + 0.05*(i-50)) 
    y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross + 0.05*(i-50)])
    Y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))
Y = Y - max(Y)
y = y - max(y)
plt.plot(x,np.exp(y))
plt.plot(x, np.exp(Y))
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/deltaphi.png")
plt.clf()




"""
exit(-1)

solution = GaiaModelPyMultiNest(changing_star_positions, star_positions, measurement_times, sigma, n_dims=ndim,
                                        n_live_points=nlive, evidence_tolerance=tol, outputfiles_basename = '/home/isabeau/Documents/Cours/isabeaugaiaGWproject/delta_results/run1');