import numpy as np


def gen_rand_point(): 
    pi = np.random.normal(0, 1, 3)
    modulus_sqr_pi = np.dot(pi,pi)
    return pi/np.sqrt(modulus_sqr_pi)
