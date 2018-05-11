import numpy as np
import matplotlib.pyplot as plt
from derivatives import *


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


