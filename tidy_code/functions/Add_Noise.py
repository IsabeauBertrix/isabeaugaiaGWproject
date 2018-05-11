# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:51:28 2018

@author: isabeau
"""
import numpy as np
import random as rd


def noise_single_star(n, t, sigma):
    x = np.array([rd.normalvariate(0,sigma), rd.normalvariate(0,sigma), rd.normalvariate(0,sigma)])
    x = x - np.dot(x, n) 
    return x


def noise(star_positions, measurement_times, sigma):
        return np.array([[noise_single_star(star_positions[i], t, sigma) for i in range(len(star_positions))] for t in measurement_times])

    
def inject_fake_noise( timing_residuals , sigma_t ):
   
    for j in range( len( timing_residuals ) ): # loop over stars
        for i in range( len( timing_residuals[j] ) ): # loop of measurements of each star
            
            delta_t = np.random.normal(0, sigma_t) # sigma_t is in nanoseconds
            timing_residuals[j][i] = timing_residuals[j][i] + delta_t
    return timing_residuals
