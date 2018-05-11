# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:53:47 2018

@author: isabeau
"""

def orthographic_projection_north(p):
    if p[2]>0:
        return [p[0], p[1]]
    else:
        return [None, None]