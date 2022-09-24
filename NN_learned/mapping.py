import numpy as np
import matplotlib.pyplot as plt
import TiO2_NN_beta as TiO2NN


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

pltNormal()
fig, axs = plt.subplots(1,1,figsize=(5,5))
axNormal(axs)

##------------------------------------------------

calculator=TiO2NN.TiO2() # generate class

T=1200       # temperature
dp1=10      # 1st TiO2 cluster diameter in angstrom
dp2=18      # 2nd TiO2 cluster diameter in angstrom

n1=calculator.DtoN(dp1) # translation from particle diameter to cluster number
n2=calculator.DtoN(dp2) # translation from particle diameter to cluster number
beta=calculator.calculateBetaNN(n1,n2,T) # main calculation
print("beta="+str(beta*1e-20)+"[m3/s]")

## Plotting results
b,v=np.meshgrid(calculator.barray,calculator.varray)
axs.set_ylabel("Initial velocity, $\it v_0$ [m/s]",size=15)
axs.set_xlabel(r'Collision parameter, $\it b$ [$\rm{\AA}$]',size=15)
axs.contourf(b,v,calculator.ps)
axs.text(100, 900, r'$\beta = {:.2f}$'.format(beta*1e15)+r'$ \times 10^{-15}$ m$^3$/s', fontsize = 12, color="white")
plt.savefig("probabilityMap.png", dpi=1000)
plt.show()
