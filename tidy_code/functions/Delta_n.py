import numpy as np

c = 2.99e8
w = np.pi/2


def delta_n ( n , t, GW_par ):
 
    # basis vectors
    epsilon_theta = np.array([GW_par.cosTheta*np.cos(GW_par.Phi), GW_par.cosTheta*np.sin(GW_par.Phi) , -np.sqrt(1-np.power(GW_par.cosTheta,2))])
    epsilon_phi = np.array([-np.sin(GW_par.Phi), np.cos(GW_par.Phi), 0])

    # direction to GW source
    q = np.array([np.sqrt(1-np.power(GW_par.cosTheta,2)) * np.cos(GW_par.Phi), np.sqrt(1-np.power(GW_par.cosTheta,2)) * np.sin(GW_par.Phi),GW_par.cosTheta])
    
    # basis tensors
    epsilon_plus= np.outer(epsilon_theta, epsilon_theta) - np.outer(epsilon_phi, epsilon_phi)
    epsilon_cross= np.outer(epsilon_theta, epsilon_phi) + np.outer(epsilon_phi,epsilon_theta)

    # metric perturbation
    H = np.exp(GW_par.logAmplus) * np.cos(GW_par.DeltaPhiPlus + t*np.exp(GW_par.logGWfrequency)) * epsilon_plus + np.exp(GW_par.logAmcross) * np.cos(GW_par.DeltaPhiCross + t*np.exp(GW_par.logGWfrequency))*epsilon_cross
    # compute astrometric deflection, delta_n
    return (n-q)/(2*(1-np.dot(q,n)))*np.dot(n,np.dot(H,n))-0.5*np.dot(H,n)
    
    
    

def delta_ncomplicated ( n , time , GW_par, distance ) :

    wWl = -2 * np.pi * distance / ( c / np.power(10 , GW_par.logGWfrequency))
    epsilon_theta = np.array([GW_par.cosTheta*np.cos(GW_par.Phi), GW_par.cosTheta*np.sin(GW_par.Phi) , -np.sqrt(1-np.power(GW_par.cosTheta,2))])
    epsilon_phi = np.array([-np.sin(GW_par.Phi), np.cos(GW_par.Phi), 0])

    q = np.array([np.sqrt(1-np.power(GW_par.cosTheta,2)) * np.cos(GW_par.Phi), np.sqrt(1-np.power(GW_par.cosTheta,2)) * np.sin(GW_par.Phi),GW_par.cosTheta])

    epsilon_plus= np.outer(epsilon_theta, epsilon_theta) - np.outer(epsilon_phi, epsilon_phi)
    epsilon_cross= np.outer(epsilon_theta, epsilon_phi) + np.outer(epsilon_phi,epsilon_theta)

    H = (np.exp(GW_par.logAmplus)*np.exp(1j*GW_par.DeltaPhiPlus)*epsilon_plus + np.exp(GW_par.logAmcross)*np.exp(1j*GW_par.DeltaPhiCross)*epsilon_cross )

    Bigterm = (1 - np.exp(-1j*wWl*(1-np.dot(q,n))))

    Hterm = ( np.dot(n,np.dot(H,n)) ) / ( 2 * (1-np.dot(q,n)) )

    denom = wWl * (1-np.dot(q,n))
    
    first = (1+(1j*(2-np.dot(q,n))/(denom)) * Bigterm) * n
    
    second = ( 1 + 1j * ( Bigterm / denom ) ) * q 

    third = (0.5 + 1j * (Bigterm / denom ) ) * np.dot(H , n) 
    
    return np.real(((first + second)*Hterm -third)*np.exp(-1j*np.exp(GW_par.logGWfrequency)*time)) 

    """
    salut = 2-np.dot(q,n)/(denom)

    first = 1+(1j* np.dot(salut , Bigterm)) * n
    
    cava = Bigterm / denom
        
    second = np.dot(cava , q ) #( 1 + 1j * cava ) * q 
    
    third = (0.5 + 1j * cava ) * np.dot(H , n) 
    
    return np.real(((first + second)*Hterm -third)*np.exp(-1j*np.exp(GW_par.logGWfrequency)*time)) 
   """
