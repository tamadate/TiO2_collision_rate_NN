import numpy as np
from mpl_toolkits import mplot3d
import TiO2_NN_beta as TiO2NN
import matplotlib.pyplot as plt

from matplotlib import ticker
niceMathTextForm = ticker.ScalarFormatter(useMathText=True)

##  To organize figure style
def pltNormal():
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['figure.subplot.bottom'] = 0.15
    plt.rcParams['figure.subplot.left'] = 0.15
    plt.rcParams["font.size"]=10

def axNormal(ax):
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

fig = plt.figure()
axs = plt.axes(projection='3d')
axs.w_zaxis.set_major_formatter(niceMathTextForm)

##------------------------------------------------

calculator=TiO2NN.TiO2()    # generate class

T=300
dps=np.arange(5,30,1)
Ndp=np.size(dps)
betas=np.zeros((Ndp,Ndp))
etas=np.zeros((Ndp,Ndp))
for i in np.arange(Ndp):
    dpi=dps[i]
    for j in np.arange(i,Ndp):
        dpj=dps[j]
        ni=calculator.DtoN(dpi)         # dp to N (1st cluster)
        nj=calculator.DtoN(dpj)         # dp to N (2nd cluster)
        beta=calculator.calculateBetaNN(ni,nj,T)    # main calculation
        L=(dpi+dpj)*0.5
        v0=(8*8.314*T/calculator.mij/np.pi)**0.5
        eta=beta/(L*L*np.pi*v0)*1e20
        betas[i][j]=beta
        betas[j][i]=beta
        etas[i][j]=etas[j][i]=eta

X, Y = np.meshgrid(dps*0.1, dps*0.1)
#axs.contour3D(X, Y, betas, 50, cmap='binary')
axs.plot_surface(X, Y, betas, cmap='rainbow', edgecolor='none')
axs.view_init(40, -100)
plt.savefig("betaMap"+str(int(T))+".png", dpi=1000)
plt.show()

X, Y = np.meshgrid(dps*0.1, dps*0.1)
fig = plt.figure()
axs = plt.axes(projection='3d')
axs.w_zaxis.set_major_formatter(niceMathTextForm)
axs.plot_surface(X, Y, etas, cmap='rainbow', edgecolor='none')
axs.view_init(40, -100)
plt.savefig("enhanceMap"+str(int(T))+".png", dpi=1000)
plt.show()
