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
fig, axs = plt.subplots(1,1,figsize=(6,5))
axNormal(axs)

##------------------------------------------------

calculator=TiO2NN.TiO2() # generate class

T=1200       # temperature
dp1=10      # 1st TiO2 cluster diameter in angstrom
dp2=10      # 2nd TiO2 cluster diameter in angstrom

strs=["0.6","1","2"]
MD=np.zeros(0)
NN=np.zeros(0)
v=np.zeros(0)
for s in strs:
    data=np.loadtxt("../MD_postProcessing/"+s+"nm.dat")
    for d in data:
        for b in np.arange(1,100,0.1):
            p=calculator.calculatePNN(d[0],d[1],b,d[2])[0]
            if(p==0.0):
                break
        MD=np.append(MD,d[3])
        NN=np.append(NN,b+1)
        v=np.append(v,d[2])
MD2=MD*MD
NN2=NN*NN
MDNN=MD*NN
N=np.size(MD)
X=N*np.sum(MD2)-(np.sum(MD))**2
Y=N*np.sum(NN2)-(np.sum(NN))**2
R2=(N*np.sum(MDNN)-np.sum(MD)*np.sum(NN))/(X**0.5*Y**0.5)
MSE=np.sum((MD-NN)**2)/N
print(R2)
print(MSE)
x=np.arange(10,81)
#axs.fill_between(x*0.1,x*0.075,x*0.133,color="gray",alpha=0.1)
axs.fill_between(x*0.1,x*0.1-1,x*0.1+1,color="gray",alpha=0.1,lw=0.01)
mainplt=axs.scatter(MD*0.1,NN*0.1,s=5,c=v,marker="o",cmap="rainbow")
axs.plot(x*0.1,x*0.1,linewidth=0.5,c="black")
axs.set_xlim([1,8])
axs.set_ylim([1,8])
axs.set_xlabel(r"$\it b_{c,MD}$ [nm]", fontsize=12)
axs.set_ylabel(r"$\it b_{c,NN}$ [nm]", fontsize=12)
axs.text(1.5,7,r"$\it R^{2}$ = "+'{:.4f}'.format(R2))
cb=fig.colorbar(mainplt)
cb.set_label(r'$v_0$ [m/s]')
plt.savefig("MDNN.png", dpi=1000)
plt.show()
