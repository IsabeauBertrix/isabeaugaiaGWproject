#         
def TestLogLikelihood(data, sky_positions, measurement_times, sigma, cube):
      
        sigmasq = sigma * sigma
        logsigma = np.log(sigma)
        GW_par = GW_parameters( logGWfrequency = cube[0], logAmplus = cube[1], logAmcross = cube[2], cosTheta = cube[3], Phi = cube[4], DeltaPhiPlus = cube[5] , DeltaPhiCross = cube[6] )


        # calculate the model
        model_sky_positions = np.array([ [ delta_n(sky_positions[i], t, GW_par) for i in range(number_of_stars)] for t in measurement_times] )


        logl = 0
        for i in range(number_of_stars):
            for j in range(len(measurement_times)):
                x = model_sky_positions[j][i] - data[j][i] 
                logl = logl - (0.5 * np.dot(x,x)/sigmasq + LN2PI + 2 * logsigma ) 
          
        return logl   

nlive = 10 #1024 #number of live points
ndim = 7 #number of parameters (n and c here)
tol = 0.5 #stopping criteria, smaller longer but more accurate


y = []
x = []
for i in range(100):
    cube = np.array([GW_par.logGWfrequency + 0.05*(i-50), GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross])
    x.append(GW_par.logGWfrequency + 0.05*(i-50)) 
    y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))

y = y - max(y)
plt.plot(x,np.exp(y))
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/gwfrequency.png")
plt.clf()

y = []
Y = []
x = []
X = []
for i in range(100):
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus + 0.05*(i-50), GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross])
    x.append(GW_par.logAmplus + 0.05*(i-50)) 
    X.append(GW_par.logAmcross + 0.05*(i-50))
    y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross + 0.05*(i-50), GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross])
    Y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))
Y = Y - max(Y)
y = y - max(y)
plt.plot(x,np.exp(y))
plt.plot(X, np.exp(Y))
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/amplitude.png")
plt.clf()

y = []
x = []
for i in range(100):
    cube = np.array([GW_par.logGWfrequency , GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta + 0.008*(i-50), GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross])
    x.append(GW_par.cosTheta + 0.008*(i-50)) 
    y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))

y = y - max(y)
plt.plot(x,np.exp(y))
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/costheta.png")
plt.clf()

y = []
x = []
for i in range(100):
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi + 0.05*(i-50), GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross])
    x.append(GW_par.Phi + 0.05*(i-50)) 
    y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))

y = y - max(y)
plt.plot(x,np.exp(y))
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/phi.png")
plt.clf()


y = []
Y = []
x = []
X = []
for i in range(100):
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus + 0.05*(i-50), GW_par.DeltaPhiCross])
    x.append(GW_par.DeltaPhiPlus + 0.05*(i-50)) 
    X.append(GW_par.DeltaPhiPlus + 0.05*(i-50)) 
    y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))
    cube = np.array([GW_par.logGWfrequency, GW_par.logAmplus, GW_par.logAmcross, GW_par.cosTheta, GW_par.Phi, GW_par.DeltaPhiPlus, GW_par.DeltaPhiCross + 0.05*(i-50)])
    Y.append(TestLogLikelihood(changing_star_positions, star_positions, measurement_times, sigma, cube))
Y = Y - max(Y)
y = y - max(y)
plt.plot(x,np.exp(y))
plt.plot(X, np.exp(Y))
plt.savefig("/home/isabeau/Documents/Cours/isabeaugaiaGWproject/deltaphi.png")
plt.clf()

