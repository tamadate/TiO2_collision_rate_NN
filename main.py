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

calculator=TiO2NN.TiO2()    # generate class

ddp=1   # Step of cluster size [angstrom]
T=300   # Temperature
f=open(str(int(T))+".dat", "w")     # output file
for dp1 in np.arange(1,30,ddp):         # 1st cluster loop
    for dp2 in np.arange(dp1,30,ddp):   # 2nd cluster loop
        n1=calculator.DtoN(dp1)         # dp to N (1st cluster)
        n2=calculator.DtoN(dp2)         # dp to N (2nd cluster)
        beta=calculator.calculateBetaNN(n1,n2,T)    # main calculation
        f.write(str(beta*1e14)+"\t")
    f.write("\n")
f.close()
