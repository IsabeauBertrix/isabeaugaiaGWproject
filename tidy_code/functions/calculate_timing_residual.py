import numpy as np  
from Delta_t import *  
from Delta_n import *
    
def calculate_timing_residuals_simple( star_positions_times_angles , GW_par ):

    x = []
    for j in range( len( star_positions_times_angles ) ): # loop over stars
        n = star_positions_times_angles[j][0]
        x.append( [ 1.0e9 * calculate_delta_t_simple(n, 1.0e-9 * star_positions_times_angles[j][1][i], star_positions_times_angles[j][2][i], GW_par) for i in range(len(star_positions_times_angles[j][1])) ] ) # loop of measurements of each star
    
    return np.array( x )

def calculate_timing_residuals_complicated ( star_positions_times_angles , GW_par, distances ):

    x = []
    for j in range( len( star_positions_times_angles ) ): # loop over stars
        n = star_positions_times_angles[j][0]
        x.append( [ 1.0e9 * calculate_delta_t_complicated(n, 1.0e-9 * star_positions_times_angles[j][1][i], star_positions_times_angles[j][2][i], GW_par, distances[j]) for i in range(len(star_positions_times_angles[j][1])) ] ) # loop of measurements of each star
    
    return np.array( x )
      

