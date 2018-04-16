# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:40:04 2018

@author: isabeau
"""

import numpy as np
import random as rd
import pylab as pl
import matplotlib.pyplot as plt
import re
import os
from sys import version

from collections import namedtuple
 
c = 2.99e8
w = np.pi/2
   
def gen_rand_point(): 
    pi = np.random.normal(0, 1, 3)
    modulus_sqr_pi = np.dot(pi,pi)
    return pi/np.sqrt(modulus_sqr_pi)
    
def LoadData( filename ):

    if ( os.path.isfile( filename ) == False ):
        print "Error: file does not exist"
        return 0

    with open( filename ) as f:
        content = f.readlines()

    data = []

    for i in range( 10 ) : #len( content ) ):

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
    
def delta_n ( n , t, GW_par ):
    epsilon_theta = np.array([GW_par.cosTheta*np.cos(GW_par.Phi), GW_par.cosTheta*np.sin(GW_par.Phi) , -np.sqrt(1-np.power(GW_par.cosTheta,2))])
    epsilon_phi = np.array([-np.sin(GW_par.Phi), np.cos(GW_par.Phi), 0])
    q = np.array([np.sqrt(1-np.power(GW_par.cosTheta,2)) * np.cos(GW_par.Phi), np.sqrt(1-np.power(GW_par.cosTheta,2)) * np.sin(GW_par.Phi),GW_par.cosTheta])
    epsilon_plus= np.outer(epsilon_theta, epsilon_theta) - np.outer(epsilon_phi, epsilon_phi)
    epsilon_cross= np.outer(epsilon_theta, epsilon_phi) + np.outer(epsilon_phi,epsilon_theta)

    H = (np.exp(GW_par.logAmplus)*np.exp(1j*GW_par.DeltaPhiPlus)*epsilon_plus + np.exp(GW_par.logAmcross)*np.exp(1j*GW_par.DeltaPhiCross)*epsilon_cross )*np.exp(1j*t*np.exp(GW_par.logGWfrequency))
    
    return np.real((n-q)/(2*(1-np.dot(q,n)))*np.dot(n,np.dot(H,n))-0.5*np.dot(H,n))
    
def delta_ncomplicated ( n , time , GW_par, distance ) :

    wWl = -2 * np.pi * distance / ( c / np.power(10 , GW_par.logGWfrequency))
    epsilon_theta = np.array([GW_par.cosTheta*np.cos(GW_par.Phi), GW_par.cosTheta*np.sin(GW_par.Phi) , -np.sqrt(1-np.power(GW_par.cosTheta,2))])
    epsilon_phi = np.array([-np.sin(GW_par.Phi), np.cos(GW_par.Phi), 0])

    q = np.array([np.sqrt(1-np.power(GW_par.cosTheta,2)) * np.cos(GW_par.Phi), np.sqrt(1-np.power(GW_par.cosTheta,2)) * np.sin(GW_par.Phi),GW_par.cosTheta])

    epsilon_plus= np.outer(epsilon_theta, epsilon_theta) - np.outer(epsilon_phi, epsilon_phi)
    epsilon_cross= np.outer(epsilon_theta, epsilon_phi) + np.outer(epsilon_phi,epsilon_theta)

    H = (np.exp(GW_par.logAmplus)*np.exp(1j*GW_par.DeltaPhiPlus)*epsilon_plus + np.exp(GW_par.logAmcross)*np.exp(1j*GW_par.DeltaPhiCross)*epsilon_cross )

    Bigterm = (1 - np.exp(-1j*wWl*(1-np.dot(q,n))))

    Hterm = ( np.dot(n,np.dot(H,n)) ) / ( 2 * (1-np.dot(q,n)) )

    denom = wWl * (1-np.dot(q,n))
    
    first = (1+(1j*(2-np.dot(q,n))/(denom)) * Bigterm) * n
    
    second = ( 1 + 1j * ( Bigterm / denom ) ) * q 

    third = (0.5 + 1j * (Bigterm / denom ) ) * np.dot(H , n) 
    
    return np.real(((first + second)*Hterm -third)*np.exp(-1j*np.exp(GW_par.logGWfrequency)*time)) 

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
        
def cartesian_coordinate_from_latitude_and_longitude(l,b):
    theta = np.pi/2. - b * np.pi / 180.
    phi = l * np.pi / 180.
    x =  np.sin(theta) * np.cos(phi)
    y =  np.sin(phi) * np.sin(theta)
    z =  np.cos(theta)
    return np.array([x,y,z])

def calculate_delta_t_complicated(n, t, psi, GW_par, distances):
 
    # spherical polar coordinates of the star
    phi = np.arctan2(n[1],n[0])
    theta = np.arccos(n[2])
    
    # basis vectors on the sky
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    e_theta = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    
    # direction of Gaia scan
    x = np.sin(psi)*e_phi - np.cos(psi)*e_theta
    
    # astrometric deflection
    dn = delta_ncomplicated ( n , t, GW_par, distances )
    
    # gaia's angular rotation frequency (rad/s) period = 6 hours
    w = 2. * np.pi / ( 6. * 60. * 60. )
    
    # compute the time delay
    return np.dot(dn,x) / w
    #IT IS IN SECONDS  
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

def calculate_timing_residuals_complicated ( star_positions_times_angles , GW_par, distances ):

    x = []
    for j in range( len( star_positions_times_angles ) ): # loop over stars
        n = star_positions_times_angles[j][0]
        x.append( [ 1.0e9 * calculate_delta_t_complicated(n, 1.0e-9 * star_positions_times_angles[j][1][i], star_positions_times_angles[j][2][i], GW_par, distances) for i in range(len(star_positions_times_angles[j][1])) ] ) # loop of measurements of each star
    
    return np.array( x )
      
GW_parameters = namedtuple("GW_parameters", "logGWfrequency logAmplus logAmcross cosTheta Phi DeltaPhiPlus DeltaPhiCross")
GW_par = GW_parameters( logGWfrequency = np.log( 2 ) - 7 * np.log(10), logAmplus = np.log(3) - 14*np.log(10), logAmcross = np.log(3) - 14*np.log(10), cosTheta = 0.5, Phi = 1.0, DeltaPhiPlus = 1*np.pi , DeltaPhiCross = np.pi/2. )
    


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

def derivative2(n , t , GW_par, param_index, scale, distance ):
    deltas = [np.power( 10 , -10.5), np.power(10, -4.75) ,np.power(10, -4.75), np.power(10 , -5.0), np.power(10 , -5.25), np.power( 10 , -4.99), np.power(10 , -4.99)]
    
    if param_index == 0:
        GW = GW_par._asdict()
        GW['logGWfrequency'] = GW['logGWfrequency'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = delta_ncomplicated( n , t , GW, distance )
        GW = GW_par._asdict()
        GW['logGWfrequency'] = GW['logGWfrequency'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - delta_ncomplicated( n, t, GW, distance)
        return answer / (2 * deltas[param_index] * scale) 
    elif param_index == 1:
        GW = GW_par._asdict()
        GW['logAmplus'] = GW['logAmplus'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = delta_ncomplicated( n , t , GW, distance )
        GW = GW_par._asdict()
        GW['logAmplus'] = GW['logAmplus'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - delta_ncomplicated( n, t, GW, distance)
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 2:
        GW = GW_par._asdict()
        GW['logAmcross'] = GW['logAmcross'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = delta_ncomplicated( n , t, GW, distance )
        GW = GW_par._asdict()
        GW['logAmcross'] = GW['logAmcross'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - delta_ncomplicated( n, t, GW, distance)
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 3:
        GW = GW_par._asdict()
        GW['cosTheta'] = GW['cosTheta'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = delta_ncomplicated( n , t , GW, distance )
        GW = GW_par._asdict()
        GW['cosTheta'] = GW['cosTheta'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - delta_ncomplicated( n, t, GW, distance)
        return answer / (2 * deltas[param_index] * scale) 
    elif param_index == 4:
        GW = GW_par._asdict()
        GW['Phi'] = GW['Phi'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = delta_ncomplicated( n , t , GW, distance )
        GW = GW_par._asdict()
        GW['Phi'] = GW['Phi'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - delta_ncomplicated( n, t, GW, distance)
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 5:
        GW = GW_par._asdict()
        GW['DeltaPhiPlus'] = GW['DeltaPhiPlus'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = delta_ncomplicated( n , t , GW, distance )
        GW = GW_par._asdict()
        GW['DeltaPhiPlus'] = GW['DeltaPhiPlus'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - delta_ncomplicated( n, t, GW, distance)
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 6:
        GW = GW_par._asdict()
        GW['DeltaPhiCross'] = GW['DeltaPhiCross'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = delta_ncomplicated( n , t , GW, distance )
        GW = GW_par._asdict()
        GW['DeltaPhiCross'] = GW['DeltaPhiCross'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - delta_ncomplicated( n, t, GW, distance)
        return answer / (2 * deltas[param_index] * scale) 
    else:
        print('error')
        return(-1)
        
def derivative3( n , t , psi, GW_par, param_index, scale ):
    deltas = [np.power( 10 , -10.5), np.power(10, -5.75) ,np.power(10, -5.75), np.power(10 , -6.25), np.power(10 , -6.25), np.power( 10 , -2.5), np.power(10 , -2.5)]
    
    if param_index == 0:
        GW = GW_par._asdict()
        GW['logGWfrequency'] = GW['logGWfrequency'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = calculate_delta_t_simple( n , t , psi, GW )
        GW = GW_par._asdict()
        GW['logGWfrequency'] = GW['logGWfrequency'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - calculate_delta_t_simple( n, t, psi, GW)
        return answer / (2 * deltas[param_index] * scale) 
    elif param_index == 1:
        GW = GW_par._asdict()
        GW['logAmplus'] = GW['logAmplus'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = calculate_delta_t_simple( n , t , psi, GW )
        GW = GW_par._asdict()
        GW['logAmplus'] = GW['logAmplus'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - calculate_delta_t_simple( n, t, psi, GW)
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 2:
        GW = GW_par._asdict()
        GW['logAmcross'] = GW['logAmcross'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = calculate_delta_t_simple( n , t , psi, GW )
        GW = GW_par._asdict()
        GW['logAmcross'] = GW['logAmcross'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - calculate_delta_t_simple( n, t, psi, GW)
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 3:
        GW = GW_par._asdict()
        GW['cosTheta'] = GW['cosTheta'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = calculate_delta_t_simple( n , t , psi, GW )
        GW = GW_par._asdict()
        GW['cosTheta'] = GW['cosTheta'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - calculate_delta_t_simple( n, t, psi, GW)
        return answer / (2 * deltas[param_index] * scale) 
    elif param_index == 4:
        GW = GW_par._asdict()
        GW['Phi'] = GW['Phi'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = calculate_delta_t_simple( n , t , psi, GW )
        GW = GW_par._asdict()
        GW['Phi'] = GW['Phi'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - calculate_delta_t_simple( n, t, psi, GW)
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 5:
        GW = GW_par._asdict()
        GW['DeltaPhiPlus'] = GW['DeltaPhiPlus'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = calculate_delta_t_simple( n , t , psi, GW )
        GW = GW_par._asdict()
        GW['DeltaPhiPlus'] = GW['DeltaPhiPlus'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - calculate_delta_t_simple( n, t, psi, GW)
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 6:
        GW = GW_par._asdict()
        GW['DeltaPhiCross'] = GW['DeltaPhiCross'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = calculate_delta_t_simple( n , t , psi, GW)
        GW = GW_par._asdict()
        GW['DeltaPhiCross'] = GW['DeltaPhiCross'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - calculate_delta_t_simple( n, t, psi, GW)
        return answer / (2 * deltas[param_index] * scale) 
    else:
        print('error')
        return(-1)
        

def derivative4( n , t , psi, GW_par, param_index, scale, distance ):
    deltas = [np.power( 10 , -10.5), np.power(10, -5.75) ,np.power(10, -5.75), np.power(10 , -6.25), np.power(10 , -6.25), np.power( 10 , -2.5), np.power(10 , -2.5)]
    
    if param_index == 0:
        GW = GW_par._asdict()
        GW['logGWfrequency'] = GW['logGWfrequency'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = calculate_delta_t_complicated( n , t , psi, GW, distance )
        GW = GW_par._asdict()
        GW['logGWfrequency'] = GW['logGWfrequency'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - calculate_delta_t_complicated( n, t, psi, GW, distance )
        return answer / (2 * deltas[param_index] * scale) 
    elif param_index == 1:
        GW = GW_par._asdict()
        GW['logAmplus'] = GW['logAmplus'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = calculate_delta_t_complicated( n , t , psi, GW, distance )
        GW = GW_par._asdict()
        GW['logAmplus'] = GW['logAmplus'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - calculate_delta_t_complicated( n, t, psi, GW, distance )
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 2:
        GW = GW_par._asdict()
        GW['logAmcross'] = GW['logAmcross'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = calculate_delta_t_complicated( n , t , psi, GW, distance )
        GW = GW_par._asdict()
        GW['logAmcross'] = GW['logAmcross'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - calculate_delta_t_complicated( n, t, psi, GW, distance )
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 3:
        GW = GW_par._asdict()
        GW['cosTheta'] = GW['cosTheta'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = calculate_delta_t_complicated( n , t , psi, GW, distance )
        GW = GW_par._asdict()
        GW['cosTheta'] = GW['cosTheta'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - calculate_delta_t_complicated( n, t, psi, GW, distance )
        return answer / (2 * deltas[param_index] * scale) 
    elif param_index == 4:
        GW = GW_par._asdict()
        GW['Phi'] = GW['Phi'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = calculate_delta_t_complicated( n , t , psi, GW, distance )
        GW = GW_par._asdict()
        GW['Phi'] = GW['Phi'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - calculate_delta_t_complicated( n, t, psi, GW, distance )
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 5:
        GW = GW_par._asdict()
        GW['DeltaPhiPlus'] = GW['DeltaPhiPlus'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = calculate_delta_t_complicated( n , t , psi, GW, distance )
        GW = GW_par._asdict()
        GW['DeltaPhiPlus'] = GW['DeltaPhiPlus'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - calculate_delta_t_complicated( n, t, psi, GW, distance )
        return answer / (2 * deltas[param_index] * scale ) 
    elif param_index == 6:
        GW = GW_par._asdict()
        GW['DeltaPhiCross'] = GW['DeltaPhiCross'] + deltas[param_index] * scale
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = calculate_delta_t_complicated( n , t , psi, GW, distance )
        GW = GW_par._asdict()
        GW['DeltaPhiCross'] = GW['DeltaPhiCross'] - deltas[param_index] * scale 
        GW = namedtuple( "GW_parameters" , GW.keys() )(**GW)
        answer = answer - calculate_delta_t_complicated( n, t, psi, GW, distance )
        return answer / (2 * deltas[param_index] * scale) 
    else:
        print('error')
        return(-1)
        
def test_derivatives1(GW_par) :
    scale_values = np.power( 10 , np.linspace(-2 , 2 , 100) )
    
    for i in range(7):
        y = [derivative1( np.array([0 , 0 , 1]), 3600 * 24 * 7 * 1.0e9 , GW_par , i , s) for s in scale_values] 
        ysq = [ np.dot(Y , Y) for Y in y] 
        plt.plot( np.log10(scale_values ) , np.log10( ysq )  )
        plt.show()
        plt.clf()

def test_derivatives2(GW_par, distance) :
    scale_values = np.power( 10 , np.linspace(-2 , 2 , 100) )
    
    for i in range(7):
        y = [derivative2( np.array([np.sin(1.0)*np.cos(1.0) , np.sin(1.0) * np.sin(1.0) , np.cos(1.0)]), 3600 * 24 * 7 * 1.0e9 , GW_par , i , s, distance) for s in scale_values] 
        ysq = [ np.dot(Y , Y) for Y in y] 
        plt.plot( np.log10(scale_values ) , np.log10( ysq )  )
        plt.show()
        plt.clf()

def test_derivatives3(GW_par) :
    scale_values = np.power( 10 , np.linspace(-2 , 2 , 100) )
    
    for i in range(7):
        y = [derivative3( np.array([0 , 0 , 1]), 3600 * 24 * 7 * 1.0e9 , np.pi/3. , GW_par , i , s) for s in scale_values] 
        ysq = [ Y * Y for Y in y] 
        plt.plot( np.log10(scale_values ) , np.log10( ysq )  )
        plt.show()
        plt.clf()

    

def test_derivatives4(GW_par, distance) :
    scale_values = np.power( 10 , np.linspace(-2 , 2 , 100) )
    
    for i in range(7):
        y = [derivative4( np.array([np.sin(1.0)*np.cos(1.0) , np.sin(1.0) * np.sin(1.0) , np.cos(1.0)]), 3600 * 24 * 7 * 1.0e9 , np.pi/3. , GW_par , i , s, distance) for s in scale_values] 
        ysq = [ Y * Y for Y in y]
        plt.plot( np.log10(scale_values ) , np.log10( ysq )  )
        plt.show()
        plt.clf()


def matrix_derivative1(n , t , GW_par):
    v1 = [derivative1( n , t , GW_par, param_index, 1.0 ) for param_index in range(7)]
    u1 = np.zeros((7 , 7))
    for i in range ( 7 ):
        for j in range ( 7 ):
            u1[i][j] = np.dot( v1[i] , v1[j] )
            
    return u1
      
    
def matrix_derivative2(n , t , GW_par, distance):
    v2 = [derivative2( n , t , GW_par, param_index, 1.0, distance ) for param_index in range(7)]
    u2 = np.zeros((7 , 7))
    for i in range ( 7 ):
        for j in range ( 7 ):
            u2[i][j] = np.dot( v2[i] , v2[j] )
            
    return u2
    
def matrix_derivative3(n , t , psi, GW_par ):
    v3 = [derivative3( n , t , np.pi/3. , GW_par, param_index, 1.0 ) for param_index in range(7)]
    return np.outer( v3 , v3 )
      
    

def matrix_derivative4(n , t , psi, GW_par, distance):
    v4 = [derivative4( n , t , np.pi/3. , GW_par, param_index, 1.0, distance ) for param_index in range(7)]
    return np.outer( v4 , v4 )
      
    

def fisher_matrix1 (star_positions_times_angles , GW_par, sigma):
    number_of_stars = len(star_positions_times_angles)
    Sigma1 = np.zeros(( 7 , 7 ))
    for i in range( number_of_stars):
        for j in range( len(star_positions_times_angles[i][2])): #len of the angles
            M = matrix_derivative1( star_positions_times_angles[i][0], star_positions_times_angles[i][1][j] * 1.0e-9 , GW_par)
            Sigma1 = Sigma1 + M / (sigma * sigma )
    return(Sigma1)            

def fisher_matrix2 (star_positions_times_angles , GW_par, sigma, distances):
    number_of_stars = len(star_positions_times_angles)
    Sigma2 = np.zeros(( 7 , 7 ))
    for i in range( number_of_stars ):
        for j in range( len(star_positions_times_angles[i][2])  ): #len of the angles
            M = matrix_derivative2( star_positions_times_angles[i][0], star_positions_times_angles[i][1][j] * 1.0e-9 , GW_par, distances[i])
            Sigma2 = Sigma2 + M / (sigma * sigma )
    return(Sigma2)
    
def fisher_matrix3 (star_positions_times_angles , GW_par, sigma_t):
    number_of_stars = len(star_positions_times_angles)
    Sigma3 = np.zeros(( 7 , 7 ))
    for i in range( number_of_stars):
        for j in range( len(star_positions_times_angles[i][1])): #len of the times
            M = matrix_derivative3( star_positions_times_angles[i][0], star_positions_times_angles[i][1][j] * 1.0e-9 , star_positions_times_angles[i][2][j], GW_par)
            Sigma3 = Sigma3 + M / (sigma_t * sigma_t )
    return(Sigma3)
    
def fisher_matrix4 (star_positions_times_angles , GW_par, sigma_t, distances):
    number_of_stars = len(star_positions_times_angles)
    Sigma4 = np.zeros(( 7 , 7 ))
    for i in range( number_of_stars):
        for j in range( len(star_positions_times_angles[i][1])): #len of the times
            M = matrix_derivative4( star_positions_times_angles[i][0], star_positions_times_angles[i][1][j] * 1.0e-9 , star_positions_times_angles[i][2][j], GW_par, distances[i])
            Sigma4 = Sigma4 + M / (sigma_t * sigma_t )
    return(Sigma4) 
         
GW_par = GW_parameters( logGWfrequency = np.log(2*np.pi/(3*28*24*3600.)), logAmplus = -12*np.log(10), logAmcross = -12*np.log(10), cosTheta = 0.5, Phi = 1.0, DeltaPhiPlus = 1 * np.pi , DeltaPhiCross = 1 * np.pi )          
#test_derivatives1(GW_par )
#test_derivatives2(GW_par, 1.0e16 )
star_positions_times_angles = LoadData( "MockAstrometricTimingData/gwastrometry-gaiasimu-1000-randomSphere-v2.dat" )
sigma = 2.9e-13 
sigma_t = 1.0e-9 
distances = np.random.normal(3.086e16 , 1.0e13, len(star_positions_times_angles))

from numpy import linalg as LA

Sigma1 = fisher_matrix1 (star_positions_times_angles , GW_par, sigma)    
w1,v1 = LA.eigh( Sigma1 )
invSigma1 = np.dot( v1 , np.dot( np.diag(1./w1) , np.transpose(v1) )  )

Sigma2 = fisher_matrix2 (star_positions_times_angles , GW_par, sigma, distances)    
w2,v2 = LA.eigh( Sigma2 )
invSigma2 = np.dot( v2 , np.dot( np.diag(1./w2) , np.transpose(v2) )  )

Sigma3 = fisher_matrix3 (star_positions_times_angles , GW_par, sigma_t)    
w3,v3 = LA.eigh( Sigma3 )
invSigma3 = np.dot( v3 , np.dot( np.diag(1./w3) , np.transpose(v3) )  )

Sigma4 = fisher_matrix4 (star_positions_times_angles , GW_par, sigma_t, distances)    
w4,v4 = LA.eigh( Sigma4 )
invSigma4 = np.dot( v4 , np.dot( np.diag(1./w4) , np.transpose(v4) )  )


print("Welcome to Skynet industry")
print("1=angle")
print("2=timing")
print("Choose carefully!!!")
reponse = raw_input("Enter 1 or 2: ")
if reponse in ['1', '2']:
    print("a=simple version")
    print("b=complicated version")
    reponse2 = raw_input("Enter a or b:")
    if reponse == '1':
        if reponse2 == 'a':
            print('( ͡° ͜ʖ ͡°)')
            error = np.sqrt(np.diag(invSigma1))
        if reponse2 == 'b':
            print('(ง ͡ʘ ͜ʖ ͡ʘ)ง')
            error = np.sqrt(np.diag(invSigma2))
    if reponse == '2':
        if reponse2 == 'a':
            print('People assume that time is a strict progression of cause to effect, but actually from a non-linear, non-subjective viewpoint, it s more like a big ball of wibbly wobbly... time-y wimey... stuff')
            error = np.sqrt(np.diag(invSigma3))
        if reponse2 == 'b':
            print('Congrats, all the Universe, will explose')
            error = np.sqrt(np.diag(invSigma4))
elif reponse == '42':
    print('je suis un ordinateur d une si infiniment subtile complexite que la vie organique elle même fait partie de mes unités de calculs. Et vous mêmes prendrez de nouvelles formes, plus primitives, et pénétreraient dans l ordinateur pour naviguer le long des dix millions d années de son programme. Je construirai cet ordinateur pour vous, et le nommerai' )
else:
    print ("Are you kidding me???!!!")
    print("lol, try again")
print(error)
exit(-1)      
number_of_stars = 1000
star_positions = [gen_rand_point() for i in range(number_of_stars)]

day = 24 * 60 * 60.
year = 3660. * 24. * 365.25
week = 3660. * 24. * 7.
month = week * 4.
measurement_times = np.arange(0, 6*month, 1*week)

#GW_par = GW_parameters( logGWfrequency = np.log(2*np.pi/(day)), logAmplus = np.log(0.5), logAmcross = np.log(0.5), cosTheta = 1.0, Phi = 1.0, DeltaPhiPlus = 0 , DeltaPhiCross = 0 )
        
changing_star_positions = np.array([ [ delta_ncomplicated(star_positions[i], t, GW_par) for i in range(number_of_stars)] for t in measurement_times] )

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
        model_sky_positions = np.array([ [ delta_ncomplicated(self._sky_positions[i], t, GW_par) for i in range(self._number_of_stars)] for t in self._measurement_times] )


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
        model_sky_positions = np.array([ [ delta_ncomplicated(sky_positions[i], t, GW_par) for i in range(number_of_stars)] for t in measurement_times] )


        logl = 0
        for i in range(number_of_stars):
            for j in range(len(measurement_times)):
                x = model_sky_positions[j][i] - data[j][i] 
                logl = logl - (0.5 * np.dot(x,x)/sigmasq + LN2PI + 2 * logsigma ) 
          
        return logl   

nlive = 10 #1024 #number of live points
ndim = 7 #number of parameters (n and c here)
tol = 0.5 #stopping criteria, smaller longer but more accurate


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

y = []
Y = []
x = []
X = []
for i in range(100):
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus + 0.05*(i-50), GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross])
    x.append(GW_par.logAmplus + 0.05*(i-50)) 
    X.append(GW_par.logAmcross + 0.05*(i-50))
    y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross + 0.05*(i-50), GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross])
    Y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))
Y = Y - max(Y)
y = y - max(y)
plt.plot(x,np.exp(y))
plt.plot(X, np.exp(Y))
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/amplitude.png")
plt.clf()

y = []
x = []
for i in range(100):
    cube = np.array([GW_par.logGWfrequency , GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta + 0.008*(i-50), GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross])
    x.append(GW_par.cosTheta + 0.008*(i-50)) 
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
X = []
for i in range(100):
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus + 0.05*(i-50), GW_par.DeltaPhiCross])
    x.append(GW_par.DeltaPhiPlus + 0.05*(i-50)) 
    X.append(GW_par.DeltaPhiPlus + 0.05*(i-50)) 
    y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross + 0.05*(i-50)])
    Y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))
Y = Y - max(Y)
y = y - max(y)
plt.plot(x,np.exp(y))
plt.plot(X, np.exp(Y))
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/deltaphi.png")
plt.clf()

