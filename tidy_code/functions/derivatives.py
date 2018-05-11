import numpy as np
from collections import namedtuple
from Delta_n import *
from Delta_t import *

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
        
