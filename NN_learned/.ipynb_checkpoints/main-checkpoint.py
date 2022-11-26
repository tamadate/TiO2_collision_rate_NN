import numpy as np
import TiO2_NN_beta as TiO2NN

calculator=TiO2NN.TiO2()                    # generate class

T=1200                                      # temperature [K]
dpi=10                                      # 1st cluster diameter [angstrom]
dpj=10                                      # 2nd cluster diameter [angstrom]

ni=calculator.DtoN(dpi)                     # dpi to N [-]
nj=calculator.DtoN(dpj)                     # dpj to N [-]
beta=calculator.calculateBetaNN(ni,nj,T)    # collision rate coefficient [m3/s]
L=(dpi+dpj)*0.5                             # collision distance [angstrom]
v0=(8*8.314*T/calculator.mij/np.pi)**0.5    # mean thermal speed [m/s]
beta0=(L*L*np.pi*v0)*1e20                   # free molecular collision rate coefficient [m3/s]
eta=beta/beta0                              # enhancement factor [-]
