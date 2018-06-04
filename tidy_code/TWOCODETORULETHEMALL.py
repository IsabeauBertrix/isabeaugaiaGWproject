
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
import sys
from sys import version
from scipy import stats
from collections import namedtuple
from numpy import linalg as LA 

c = 2.99e8
w = np.pi/2

sys.path.append("functions/")
from gen_rand_GW import *
from LoadData import *
from Delta_n import *
from CoordinateConversion import *
from Delta_t import *
from Add_Noise import *
from derivatives import *
from calculate_timing_residual import *
from test_derivative import *
from MATRIX import *

day = 24 * 60 * 60.
year = 3660. * 24. * 365.25
week = 3660. * 24. * 7.
month = week * 4.
microarcsecond = np.pi/(180*3600*1e6)
  
GW_parameters = namedtuple("GW_parameters", "logGWfrequency logAmplus logAmcross cosTheta Phi DeltaPhiPlus DeltaPhiCross")
GW_par = gen_rand_GW ()   
#GW_par = GW_parameters( logGWfrequency = np.log(2*np.pi/(3*month)), logAmplus = -12*np.log(10), logAmcross = -12*np.log(10), cosTheta = 0.5, Phi = 1.0, DeltaPhiPlus = 1*np.pi , DeltaPhiCross = np.pi ) 

star_positions_times_angles = LoadData( "MockAstrometricTimingData/gwastrometry-gaiasimu-1000-randomSphere-v2.dat")

number_of_stars = len(star_positions_times_angles)
sigma = 100 * microarcsecond / np.sqrt ( 1.0e9 / number_of_stars )
sigma_t = 1.667 * 1.0e-6 / np.sqrt ( 1.0e9 / number_of_stars )
distances = np.random.normal(3.086e16 , 1.0e13, len(star_positions_times_angles))



def WapperFunction_FisherMatrix ( args ):
    
    sigma = args[0]
    sigma_t = args[1]
    distances = args[2]
    d = args[3]
    
    GW_par = args[4]
    
    SIGMA1 = fisher_matrix1 (d , GW_par , sigma )
    w1,v1 = LA.eigh( SIGMA1 )
    invSIGMA1 = np.dot( v1 , np.dot( np.diag(1./w1) , np.transpose(v1) )  )
    
    SIGMA2 = fisher_matrix2 (d , GW_par , sigma , distances )
    w2,v2 = LA.eigh( SIGMA2 )
    invSIGMA2 = np.dot( v2 , np.dot( np.diag(1./w2) , np.transpose(v2) )  )
        
    SIGMA3 = fisher_matrix3 (d , GW_par , sigma_t )
    w3,v3 = LA.eigh( SIGMA3 )
    invSIGMA3 = np.dot( v3 , np.dot( np.diag(1./w3) , np.transpose(v3) )  )
        
    SIGMA4 = fisher_matrix4 (d , GW_par , sigma_t , distances )
    w4,v4 = LA.eigh( SIGMA4 )
    invSIGMA4 = np.dot( v4 , np.dot( np.diag(1./w4) , np.transpose(v4) )  )
        
    return [ GW_par , invSIGMA1 , invSIGMA2 , invSIGMA3, invSIGMA4 ]
    

def Save_Results_To_File ( results , filename1, filename2, filename3, filename4 ):
    with open(filename1, 'w') as f:
        for r in results:
            f.write( str(r[0]) )
            f.write( '\n' )
            f.write( str(r[1]))
            f.write( '\n' )
    with open(filename2, 'w') as f:
        for r in results:
            f.write( str(r[0]) )
            f.write( '\n' )
            f.write( str(r[2]))
            f.write( '\n' )
    with open(filename3, 'w') as f:
        for r in results:
            f.write( str(r[0]) )
            f.write( '\n' )
            f.write( str(r[3]))
            f.write( '\n' )
    with open(filename4, 'w') as f:
        for r in results:
            f.write( str(r[0]) )
            f.write( '\n' )
            f.write( str(r[4]))
            f.write( '\n' )
    return 1
		
def Load_results_to_File (filename):
    return [logOmegaVals, logAplusVals, logAcrossVals, cosThetaVals, PhiVals, DeltaPhiPlusVals, DeltaPhiCrossVals, DeltalogOmega, DeltalogAplus, DeltaLogAcross, DeltacosTheta, DeltaPhi, DeltaDeltaPhiPlus, DeltaDeltaPhiCross]

num = 250
num_cpus = 24

from multiprocessing import Pool
p = Pool( num_cpus )

argument_list = [ [ sigma, sigma_t, distances, star_positions_times_angles, gen_rand_GW () ] for i in range(num) ]
results = p.map ( WapperFunction_FisherMatrix , argument_list )
Save_Results_To_File ( results , "test1_2.dat", "test2_2.dat", "test3_2.dat", "test4_2.dat" )

    
"""
def oneDhist_automatic_xrange( chain ):
    #kernel = stats.gaussian_kde( chain )
    hist, bin_edges = np.histogram(chain)
    #area = len(chain) * ( bin_edges[1] - bin_edges[0] )
    plt.hist(chain, bins = bin_edges)
    plt.show()
    return 1
oneDhist_automatic_xrange(matrix[:,0] )
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/fisher_matrix/logfrenquency.png")
oneDhist_automatic_xrange(matrix[:,1] )
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/fisher_matrix/logAmplus.png")
oneDhist_automatic_xrange(matrix[:,2] )
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/fisher_matrix/logAmcross.png")
oneDhist_automatic_xrange(matrix[:,3] )
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/fisher_matrix/costheta.png")
oneDhist_automatic_xrange(matrix[:,4] )
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/fisher_matrix/phi.png")
oneDhist_automatic_xrange(matrix[:,5] )
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/fisher_matrix/deltaphiplus.png")
oneDhist_automatic_xrange(matrix[:,6] )
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/fisher_matrix/deltaphicross.png*")
         

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
            reponse3 = raw_input("Oh wait, do you also want to see the plots? [yes/no]:")
            if reponse3 == 'yes':
                test_derivatives1(GW_par)
            if reponse3 == 'no':
                print('well ok... ʕ•͡-•ʔ ')
            print('( ͡° ͜ʖ ͡°)')
            error = np.sqrt(np.diag(invSigma1))
            print(error)
        if reponse2 == 'b':
            reponse3 = raw_input("Oh wait, do you also want to see the plots? [yes/no]:")
            if reponse3 == 'yes':
                test_derivatives2(GW_par, 1.0e16)
            if reponse3 == 'no':
                print('well ok... ʕ•͡-•ʔ ')
            print('(ง ͡ʘ ͜ʖ ͡ʘ)ง')
            error = np.sqrt(np.diag(invSigma2))
            print(error)
    if reponse == '2':
        if reponse2 == 'a':
            reponse3 = raw_input("Oh wait, do you also want to see the plots? [yes/no]:")
            if reponse3 == 'yes':
                test_derivatives3(invSigma3)
            if reponse3 == 'no':
                print('well ok... ʕ•͡-•ʔ ')
            print('Well...You know, people assume that time is a strict progression of cause to effect, but actually from a non-linear, non-subjective viewpoint, it s more like a big ball of wibbly wobbly... time-y wimey... stuff')
            error = np.sqrt(np.diag(invSigma3))
            print(error)
        if reponse2 == 'b':
            reponse3 = raw_input("Oh wait, do you also want to see the plots? [yes/no]:")
            if reponse3 == 'yes':
                print('Well...that s embarassing but for the moment it do not work. Please accept our excuses for the disturbance')
            if reponse3 == 'no':
                print('well ok... ʕ•͡-•ʔ ')
            print('Congrats, all the Universe, will explose')
            error = np.sqrt(np.diag(invSigma4))
            print(error)
elif reponse == '42':
    print('je suis un ordinateur d une si infiniment subtile complexite que la vie organique elle même fait partie de mes unités de calculs. Et vous mêmes prendrez de nouvelles formes, plus primitives, et pénétreraient dans l ordinateur pour naviguer le long des dix millions d années de son programme. Je construirai cet ordinateur pour vous, et le nommerai' )
    print('DO NOT FORGET YOUR TOWEL, NEVER')
    print('Alien are afraid of towel')
    print('For instance, on the planet Earth, man had always assumed that he was more intelligent than dolphins because he had achieved so much—the wheel, New York, wars and so on—whilst all the dolphins had ever done was muck about in the water having a good time. But conversely, the dolphins had always believed that they were far more intelligent than man—for precisely the same reasons.')
elif reponse == '66':
    reponse4 = raw_input("Did you ever hear the trajedy of Dark Plagueis The Wise? [yes/no]:   ")
    if reponse4 == 'yes':
        print("OMG YOU ARE THE DARK LORD OF THE SITH!!!!!!! (；☉_☉)")
        reponse5 = raw_input("Execute order 66? [yes/no]:    ")
        if reponse5 == 'yes':
            print("(✖╭╮✖)")
        if reponse5 == 'no':
            print("┌(˘⌣˘)ʃ")
    if reponse4 == 'no':
        print("I thought not. It's not a story the jedi will tell you. It's a Sith legend. Dark Plagueis was a Dark Lord of the Sith, so powerful and so wise he could use the Force to influence the midichlorians to create life ... He had such a knowledge of the dark side, he could even keep the ones he cared about from dying." )
else:
    print ("Good aswer.")
    print("Thank you for your participation, we hope you will have a great life. Please enjoy your day")
"""
