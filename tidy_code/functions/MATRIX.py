import numpy as np
from numpy import linalg as LA 
from gen_rand_GW import *
from LoadData import *
from Delta_n import *
from CoordinateConversion import *
from Delta_t import *

from derivatives import *
from calculate_timing_residual import *

from derivatives import *


GW_parameters = namedtuple("GW_parameters", "logGWfrequency logAmplus logAmcross cosTheta Phi DeltaPhiPlus DeltaPhiCross")

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
    
