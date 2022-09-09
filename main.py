import numpy as np
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

calculator=TiO2NN.TiO2()

ddp=1
T=300
dp1s=np.arange(1,30,ddp)
with open(str(int(T))+".dat", "w") as f:
    for dp1 in dp1s:
        for dp2 in np.arange(dp1,30,ddp):
            n1=calculator.DtoN(dp1)
            n2=calculator.DtoN(dp2)
            beta=calculator.calculateBetaNN(n1,n2,T)
            f.write(str(beta*1e14)+"\t")
        f.write("\n")
f.close()
