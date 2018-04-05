# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:43:45 2018

@author: isabeau
"""

import numpy as np
import os
import re

def Separator( filename ):
      
    if ( os.path.isfile( filename ) == False ):
        print "Error: file does not exist"
        return 0

    with open( filename ) as f:
        content = f.readlines()
    Times = []
    for i in range( 1000 ): #len(content)

        line = content[i]
        line = re.split(', \[|\], \[|\]',line)
        
        Time = np.array( [ np.uint64(a) for a in re.split(', ',line[1]) ] )
        Times.append([Time])
    return Times



 
Times = Separator( "MockAstrometricTimingData/gwastrometry-gaiasimu-1000-randomSphere-v2.dat" )
#print(Times[5][0][0])
#print(Times[5][0][1])
#print(Times[5][0][2])
"""
print(len(Times[5][0]))
print(len(Times[11][0]))
print(len(Times[12][0]))
"""
bitemps=np.zeros((1000,2,500), dtype='uint64')
file = open('pairs.txt', "rw")
        
for j in range(1000):
    if len(Times[j][0]) % 2 == 0:
        
        for i in range((len(Times[j][0]))/2):
            #print(i,j)
            #print(len(Times[j][0]))
            bitemps[j][0][i]=Times[j][0][i]
            bitemps[j][1][i]=Times[j][0][i+len((Times[j][0]))/2]
            #print(bitemps[j][0][i], bitemps[j][1][i])
            file.write(str(bitemps[j][0][i])) #bitemps[j][1][i])
    else:
        for i in range((len(Times[j][0])-1)/2):
            #print(i,j)
            #print(len(Times[j][0]))
            bitemps[j][0][i]=Times[j][0][i]
            bitemps[j][1][i]=Times[j][0][i+1+len((Times[j][0])-1)/2]
            #print(bitemps[j][0][i], bitemps[j][1][i]
            file.write(str(bitemps[j][0][i])) #bitemps[j][1][i])

