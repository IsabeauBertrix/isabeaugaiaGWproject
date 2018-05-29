import numpy as np
import random as rd
from collections import namedtuple

GW_parameters = namedtuple("GW_parameters", "logGWfrequency logAmplus logAmcross cosTheta Phi DeltaPhiPlus DeltaPhiCross")
def gen_rand_GW():
    
    x = np.random.uniform( -8. , -7. )
    Omega = np.power( 10. , x ) 

    log10Amplus = np.random.uniform( -15, -10 )
    log10Amcross = np.random.uniform( -15, -10 )
    A = np.sqrt( np.power(np.power(10, log10Amcross) , 2) + np.power(np.power(10, log10Amplus) , 2))
    log10Amplus = log10Amplus - 12 - np.log10(A)
    log10Amcross = log10Amcross - 12 - np.log10(A)

    cosTheta = np.random.uniform(-1 , 1 )
    Phi = np.random.uniform(0 , 2 * np.pi )
    
    DeltaPhiPlus = np.random.uniform( 0 , 2 * np.pi )
    DeltaPhiCross = np.random.uniform( 0 , 2 * np.pi )   

    GW_par = GW_parameters( logGWfrequency = np.log( Omega ) , logAmplus = log10Amplus, logAmcross = log10Amcross, cosTheta = cosTheta, Phi = Phi, DeltaPhiPlus = DeltaPhiPlus , DeltaPhiCross = DeltaPhiCross )
    return GW_par   

