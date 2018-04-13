# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:43:45 2018
@author: isabeau
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def LoadTimes( filename ):
      
    if ( os.path.isfile( filename ) == False ):
        print "Error: file does not exist"
        return 0

    with open( filename ) as f:
        content = f.readlines()
    Times = []
    for i in range( 1000 ): #len(content)

        line = content[i]
        line = re.split(', \[|\], \[|\]',line)
        
        Time = np.array( [ np.int64(a) for a in re.split(', ',line[1]) ] )
        Times.append([Time])
      
    return Times

def SplitTimes( Times ):
    answer = []
    for t in Times:
        index = 0
        for i in range( len( t[0] ) - 1 ):
            if t[0][i+1] < t[0][i]:
                index = i
        #print(t)
        first = t[0][0:index+1]
        second = t[0][index+1::]
        answer.append([first,second])
    return answer

 
Times = LoadTimes( "MockAstrometricTimingData/gwastrometry-gaiasimu-1000-randomSphere-v2.dat" )
Split = SplitTimes( Times )
#print(Times[0][0])
#print(Split[1][1][1])

matrix = []
Count = []
for i in range(1000):
    liste = []
    delay = 2 * 3600 * 1.0e9
    for j in range(len(Split[i][0])):
        count = 0
        for k in range(len(Split[i][1])):
            if np.fabs(Split[i][0][j]  - Split[i][1][k] ) < delay :
                liste.append([Split[i][0][j], Split[i][1][k]])
                count = count + 1
        Count.append(count)
    matrix.append(liste)
print(matrix[0])
plt.hist(Count, [-0.5, 0.5, 1.5, 2.5])
plt.show()
exit(-1)
#print(Times[5][0][0])
#print(Times[5][0][1])
#print(Times[5][0][2])

bitemps=np.zeros((1000,2,500), dtype='uint64')

        
for j in range(1000):
    if len(Times[j][0]) % 2 == 0:
        
        for i in range((len(Times[j][0]))/2):
            #print(i,j)
            #print(len(Times[j][0]))
            bitemps[j][0][i]=Times[j][0][i]
            bitemps[j][1][i]=Times[j][0][i+len((Times[j][0]))/2]
            #print(bitemps[j][0][i], bitemps[j][1][i])
            
    else:
        for i in range((len(Times[j][0])-1)/2):
            #print(i,j)
            #print(len(Times[j][0]))
            bitemps[j][0][i]=Times[j][0][i]
            bitemps[j][1][i]=Times[j][0][i+1+len((Times[j][0])-1)/2]
            #print(bitemps[j][0][i], bitemps[j][1][i]
