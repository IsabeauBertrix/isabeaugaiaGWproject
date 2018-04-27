import numpy as np
import os
import re

def LoadData( filename , NumberOfStarsToLoad = -1 ):

    if NumberOfStarsToLoad < 0:
        num = len ( content )
    else:
        num = NumberOfStarsToLoad

    if ( os.path.isfile( filename ) == False ):
        print ( "Error: file does not exist" )
        return 0

    with open( filename ) as f:
        content = f.readlines()

    data = []

    for i in range( num ):

        line = content[i]
        line = re.split(', \[|\], \]|\]',line)

        SkyPosition = np.array( [ float( re.split(', ',line[0])[index] ) for index in [0,1] ] )
        
        SkyPosition = cartesian_coordinate_from_latitude_and_longitude(SkyPosition[0],SkyPosition[1])       
        
        Times = np.array( [ np.uint64(a) for a in re.split(', ',line[1]) ] )

        ScanAngles = np.array( [ float(a) for a in re.split(', ',line[3]) ] )

        if ( len(Times) != len(ScanAngles) ):
            print ( "Error: something bad has happened in LoadData()" )
            return 0

        data.append( [ SkyPosition , Times , ScanAngles] )

    return data
    
