# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:55:09 2018

@author: isabeau
"""
import numpy as np

def cartesian_coordinate_from_latitude_and_longitude(l,b):
    theta = np.pi/2. - b * np.pi / 180.
    phi = l * np.pi / 180.
    x =  np.sin(theta) * np.cos(phi)
    y =  np.sin(phi) * np.sin(theta)
    z =  np.cos(theta)
    return np.array([x,y,z])