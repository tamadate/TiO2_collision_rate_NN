import matplotlib.pyplot as plt
import numpy as np
import TiO2


class plot():
    saveLoc="/home/tama3rdgen/TiO2/Paper/MDresults/"
## --------------   Plotting results   -------------- ##
    def pltNormal(self):
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams["font.size"]=12

    def axNormal(self,ax):
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')

## --------------   Probability mapping   -------------- ##
    def probabilityMap(self,MD):
        self.pltNormal()
        fig, axs = plt.subplots(3,5,figsize=(18,10))
        for i in np.arange(15):
        	self.axNormal(axs.flat[i])
        fig.text(0.55,0.1,r"Impact parameter, $b$ [nm]",ha="center",size=20)
        fig.text(0.10,0.55,"Initial velocity, $v_0$ [m/s]",va="center",rotation="vertical",size=20)
        figRes=50

        b,v=np.meshgrid(MD.bs*0.1,MD.vs)
        for n2 in np.arange(MD.Nsize):
            axs.flat[n2].text(3.5,600,r"$D_{{\rm p},j}$="+MD.dpString[n2],size=15,color="white")
            axs.flat[n2].contourf(b,v,MD.pArray[n2],figRes,cmap="rainbow")
            axs.flat[n2].axvline(x = (MD.dpSize[MD.n1]+MD.dpSize[n2])*0.5, ls='--', color = 'black')

        fig.delaxes(axs.flat[13])
        fig.delaxes(axs.flat[14])
        im1=axs.flat[12].contourf(b,v,MD.pArray[12],figRes,cmap="rainbow")
        cbar=plt.colorbar(im1, ax=axs[2,3],fraction=0.85,aspect=5,shrink=0.8,pad=2,ticks=np.array([0,0.2,0.4,0.6,0.8,1]))
        fig.text(0.68,0.26,"Probability [-]",va="center",rotation="vertical",size=15)
        fig.text(0.75, 0.25, r"$D_{{\rm p},i}$ = "+MD.dpString[MD.n1], fontsize = 20)
        plt.savefig(self.saveLoc+MD.dpString[MD.n1]+"Probability.png", dpi=1000)
        plt.show()
#*******************************************************************************
## --------------   Flag mapping   -------------- ##
    def flagMap(self,MD):
        self.pltNormal()
        fig, axs = plt.subplots(3,5,figsize=(18,10))
        for i in np.arange(15):
        	self.axNormal(axs.flat[i])
        fig.text(0.55,0.1,r"Impact parameter, $b$ [nm]",ha="center",size=20)
        fig.text(0.10,0.55,"Initial velocity, $v_0$ [m/s]",va="center",rotation="vertical",size=20)
        figRes=1

        b,v=np.meshgrid(MD.bs*0.1,MD.vs)
        for n2 in np.arange(MD.Nsize):
            axs.flat[n2].set_title(r"$D_{{\rm p},j}$="+MD.dpString[n2],loc="left",size=15)
            axs.flat[n2].contourf(b,v,MD.flagArray[n2],figRes,cmap="rainbow",levels=[0,0.5,1])

        fig.delaxes(axs.flat[13])
        fig.delaxes(axs.flat[14])
        im1=axs.flat[12].contourf(b,v,MD.flagArray[12],figRes,cmap="rainbow")
        cbar=plt.colorbar(im1, ax=axs[2,3],fraction=0.85,aspect=5,shrink=0.8,pad=2)
        fig.text(0.68,0.26,"No / Yes",va="center",rotation="vertical",size=15)
        fig.text(0.75, 0.25, r"$D_{{\rm p},i}$ = "+MD.dpString[MD.n1], fontsize = 20)
        plt.savefig(self.saveLoc+MD.dpString[MD.n1]+"Flag.png", dpi=1000)
        plt.show()
#*******************************************************************************
## --------------   Binding length mapping   -------------- ##
    def bindingLengthMap(self,MD,setT):
        self.pltNormal()
        fig, axs = plt.subplots(3,5,figsize=(18,10))
        for i in np.arange(15):
        	self.axNormal(axs.flat[i])
        fig.text(0.55,0.1,r"Impact parameter, $b$ [nm]",ha="center",size=20)
        fig.text(0.10,0.55,"Initial velocity, $v_0$ [m/s]",va="center",rotation="vertical",size=20)
        figRes=50

        b,v=np.meshgrid(MD.bs*0.1,MD.vs)
        for n2 in np.arange(MD.Nsize):
            MD.tempSet(setT)
            MD.BLmapping(n2)
            axs.flat[n2].text(3.5,600,r"$D_{{\rm p},j}$="+MD.dpString[n2],size=15,color="white")
            axs.flat[n2].contourf(b,v,MD.blArray[n2]/np.max(MD.blArray[n2]),figRes,cmap="rainbow")

        fig.delaxes(axs.flat[13])
        fig.delaxes(axs.flat[14])
        im1=axs.flat[12].contourf(b,v,MD.blArray[12]/np.max(MD.blArray[12]),figRes,cmap="rainbow")
        cbar=plt.colorbar(im1, ax=axs[2,3],fraction=0.85,aspect=5,shrink=0.8,pad=2,ticks=np.array([0,0.2,0.4,0.6,0.8,1]))
        fig.text(0.685,0.17,"Normalized\nbinding length [-]",ha="center",rotation="vertical",size=15)
        fig.text(0.75, 0.25, r"$D_{{\rm p},i}$ = "+MD.dpString[MD.n1]+"\n$T$ = "+'{:.0f}'.format(MD.T)+" K", fontsize = 20)
        plt.savefig(self.saveLoc+MD.dpString[MD.n1]+"BindingLength"+str(int(MD.T))+".png", dpi=1000)
        plt.show()
#*******************************************************************************
## --------------   Binding length mapping with different temperatures   -------------- ##
    def BLmappingTemp(self,MD,Tmin,Tmax):
        self.pltNormal()
        fig, axs = plt.subplots(2,4,figsize=(15,5),sharey=True,gridspec_kw={'width_ratios': [5, 1, 5, 1]})
        for i in np.arange(4):
        	self.axNormal(axs.flat[i])
        plt.rcParams["font.size"]=12
        fig.text(0.55,0.05,r"Impact parameter, $b$ [nm]",ha="center",size=20)
        fig.text(0.10,0.55,"Initial velocity, $v_0$ [m/s]",va="center",rotation="vertical",size=20)
        figRes=50
        b,v=np.meshgrid(MD.bs*0.1,MD.vs)
        tempArray=np.linspace(Tmin,Tmax,4)
        Tsize=np.size(tempArray)
        plt.ylim(20,400)

        n2=12
        for i in np.arange(Tsize):
            MD.tempSet(tempArray[i])
            print(MD.T)
            MD.BLmapping(n2)
            axs.flat[i*2].text(3.5,300,r"$T$ = "+str(int(MD.T))+" K\n"+"$D_{{\\rm p},i}$ = 1 nm" +"\n"+ "$D_{{\\rm p},j}$ = 3 nm",size=15,color="white")
            axs.flat[i*2].contourf(b,v,MD.blArray[n2]/np.max(MD.blArray[n2]),figRes,cmap="rainbow")
            axs.flat[i*2+1].plot(MD.blArray[n2].sum(axis=1)*1e-6,v,lineWidth=0.5,c="black")
            MB=(MD.mij*0.5/np.pi/MD.T/MD.kb)**1.5*4*np.pi*v*v*np.exp(-MD.mij*v*v/MD.T/MD.kb)
            axs.flat[i*2+1].plot(MB,v,lineWidth=0.5,c="black")

        im=axs.flat[6].contourf(b,v,MD.blArray[n2]/np.max(MD.blArray[n2]),figRes,cmap="rainbow")
        fig.text(0.97,0.22,"Normalized binding length [-]",ha="center",rotation="vertical",size=15)
        cbar_ax = fig.add_axes([0.92, 0.16, 0.01, 0.71])
        fig.colorbar(im, cax=cbar_ax,ticks=np.array([0,0.2,0.4,0.6,0.8,1]))
        plt.savefig(self.saveLoc+MD.dpString[MD.n1]+"BLtemp.png", dpi=1000)
        plt.show()
#*******************************************************************************
## --------------   Beta as function of temp.   -------------- ##
    def betaTemp(self,MD,Tlow,Thigh):
        self.pltNormal()
        fig, axs = plt.subplots(1,1,figsize=(5,5),sharey=True)
        self.axNormal(axs)
        tempArray=np.arange(Tlow,Thigh,10)
        Tsize=np.size(tempArray)

        cm=plt.get_cmap("seismic")
        for n22 in np.arange(MD.Nsize*0.333):
            n2=int(n22*3)
            betas=np.zeros(Tsize)
            for i in np.arange(Tsize):
                MD.tempSet(tempArray[i])
                betas[i]=MD.beta(n2)
            axs.plot(tempArray,betas,label=str(MD.dpString[n2]),color=cm(n22*0.2+0.2),linewidth=0.8)
        axs.set_xlim([Tlow,Thigh])
        axs.set_xlabel("Temperature [K]",size=15)
        axs.set_ylabel(r"Collision kernel, $\beta _{ij}$ [m$^3$ s$^ {-1}$]",size=15)
        plt.ticklabel_format(style='sci', axis='y', useMathText=True)
        #plt.legend()
        plt.savefig(self.saveLoc+MD.dpString[MD.n1]+"Beta.png", dpi=1000)
        plt.show()
#*******************************************************************************
## --------------   Enhancement facto as function of temp.   -------------- ##
    def enhanceTemp(self,MD,Tlow,Thigh):
        self.pltNormal()
        fig, axs = plt.subplots(1,1,figsize=(5,5),sharey=True)
        self.axNormal(axs)
        tempArray=np.arange(Tlow,Thigh,10)
        Tsize=np.size(tempArray)

        cm=plt.get_cmap("seismic")
        for n22 in np.arange(MD.Nsize*0.333):
            n2=int(n22*3)
            etas=np.zeros(Tsize)
            for i in np.arange(Tsize):
                MD.tempSet(tempArray[i])
                etas[i]=MD.eta(n2)
            axs.plot(tempArray,etas,label=str(MD.dpString[n2]),color=cm(n22*0.2+0.2),linewidth=0.8)
        axs.set_xlim([Tlow,Thigh])
        axs.set_xlabel("Temperature [K]",size=15)
        axs.set_ylabel(r"Enhancement factor, $\eta$ [-]",size=15)
        #plt.legend()
        plt.savefig(self.saveLoc+MD.dpString[MD.n1]+"Enhance.png", dpi=1000)
        plt.show()
