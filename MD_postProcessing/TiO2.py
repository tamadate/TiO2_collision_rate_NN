import numpy as np
import os
import glob

class TiO2():
## --------------   From MD simulation   -------------- ##
    loc="/media/tama3rdgen/6TB/TiO2/MD_results/"
    ns=np.array((9,24,48,84,129,198,291,390,534,696,882,1098,1356))
    dpString=np.array(("0.6nm","0.8nm","1nm","1.2nm","1.4nm","1.6nm","1.8nm","2nm","2.2nm","2.4nm","2.6nm","2.8nm","3nm"))
    dpSize=np.array((0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3))
    dv1=20.0
    dv2=20.0
    db=2.0
    vs=np.arange(20.0,210.0,20.0)
    vs=np.append(vs,np.arange(250.0,710.0,50.0))
    bs=np.arange(0.0,80.0,db)

    Nsize=np.size(ns)
    nvs=np.size(vs)
    nbs=np.size(bs)

    pArray=np.zeros(((Nsize,nvs,nbs)))
    flagArray=np.zeros(((Nsize,nvs,nbs)))
    blArray=np.zeros(((Nsize,nvs,nbs)))

## --------------   Variables and constants   -------------- ##
    T=1000.0
    kb=1.38e-23
    kbT=kb*T
    MTiO2=(48+16*2)/1000.0/6.02e23
    n1=2

## --------------   Main loop   -------------- ##
    def tempSet(self,setT):
        self.T=setT
        self.kbT=self.T*self.kb

    def sizeSet(self,size):
        if(size==1):
            self.n1=2
        if(size==2):
            self.n1=7
        if(size==3):
            self.n1=0
        if(size==4):
            self.n1=12

    def mapping(self):
        for n2 in np.arange(self.Nsize):
            if(self.n1==12):
                if (n2!=12 and n2!=0 and n2!=7 and n2!=2):
                    continue
            n=np.array((self.ns[self.n1],self.ns[n2]))       # [Ni,Nj]
            for dirname in glob.glob(self.loc+str(self.dpString[self.n1])+str(self.dpString[n2])+"*/"):
                for iv in np.arange(self.nvs):
                    filename=str(dirname)+str("{:.01f}".format(self.vs[iv]/100.0))+"/0/probability.dat"
                    if(os.path.exists(filename)):
                        data=np.loadtxt(filename)
                        for i in np.arange(1,10):
                            data+=np.loadtxt(str(dirname)+"/"+str("{:.01f}".format(self.vs[iv]/100.0))+"/"+str(i)+"/probability.dat")
                        data/=10.0
                        for ib in np.arange(self.nbs):
                            for d in data:
                                if d[0]==self.bs[ib]:
                                    self.pArray[n2][iv][ib]=d[2]
                                    self.flagArray[n2][iv][ib]=1
            ## propability calculation
            for iv in np.arange(self.nvs):
                p=1.0
                for ib in np.arange(self.nbs):
                    if (self.flagArray[n2][iv][ib]==0):
                        self.pArray[n2][iv][ib]=p
                    if (self.flagArray[n2][iv][ib]==1 and p==1.0):
                        p=0.0

    def BLmapping(self,n2):
        ## binding length calculation
        n=np.array((self.ns[self.n1],self.ns[n2]))       # [Ni,Nj]
        m=n/3.0*self.MTiO2                     # [mi,mj]
        mij=m[0]*m[1]/(m[0]+m[1])         # 1/mij=1/mi+1/mj
        self.mij=mij
        coeff=(mij*0.5/np.pi/self.kbT)**1.5*8*np.pi**2
        for iv in np.arange(self.nvs):
            dv=0.0
            if iv==0:
                dv+=10.0
            else:
                dv+=(self.vs[iv]-self.vs[iv-1])*0.5
            if iv==self.nvs-1:
                dv+=25.0
            else:
                dv+=(self.vs[iv+1]-self.vs[iv])*0.5
            for ib in np.arange(self.nbs):
                self.blArray[n2][iv][ib]=coeff*self.vs[iv]**3*np.exp(-mij*self.vs[iv]**2*0.5/self.kbT)*self.bs[ib]*self.pArray[n2][iv][ib]

    def beta(self,n2):
        ## binding length calculation
        n=np.array((self.ns[self.n1],self.ns[n2]))       # [Ni,Nj]
        m=n/3.0*self.MTiO2                     # [mi,mj]
        mij=m[0]*m[1]/(m[0]+m[1])         # 1/mij=1/mi+1/mj
        coeff=(mij*0.5/np.pi/self.kbT)**1.5*8*np.pi**2
        beta=0
        for iv in np.arange(self.nvs):
            dv=0.0
            if iv==0:
                dv+=10.0
            else:
                dv+=(self.vs[iv]-self.vs[iv-1])*0.5
            if iv==self.nvs-1:
                dv+=25.0
            else:
                dv+=(self.vs[iv+1]-self.vs[iv])*0.5
            for ib in np.arange(self.nbs):
                length=coeff*self.vs[iv]**3*np.exp(-mij*self.vs[iv]**2*0.5/self.kbT)*self.bs[ib]*self.pArray[n2][iv][ib]
                beta+=length*self.db*dv
        return beta*1e-20

    def eta(self,n2):
        beta=self.beta(n2)
        n=np.array((self.ns[self.n1],self.ns[n2]))       # [Ni,Nj]
        m=n/3.0*self.MTiO2                     # [mi,mj]
        mij=m[0]*m[1]/(m[0]+m[1])         # 1/mij=1/mi+1/mj
        vmean=(8*self.kbT/np.pi/mij)**0.5
        L=(self.dpSize[self.n1]+self.dpSize[n2])*0.5
        betaFM=np.pi*L*L*vmean*1e-18              # nm*nm*m/s
        eta=beta/betaFM
        return eta

    def critical(self):
        f=open(self.dpString[self.n1]+".dat","w")
        for n2 in np.arange(self.Nsize):
            if(self.n1==12):
                if (n2!=12 and n2!=0 and n2!=7 and n2!=2):
                    continue
            ## propability calculation
            for iv in np.arange(self.nvs):
                nocoll=0
                for ib in np.arange(self.nbs):
                    if(self.pArray[n2][iv][ib]==1.0):
                        coll=self.bs[ib]
                    if(self.pArray[n2][iv][ib]==0.0):
                        nocoll=self.bs[ib]
                        break
                if(nocoll!=0):
                    f.write(str(self.ns[self.n1])+" "+str(self.ns[n2])+" "+str(self.vs[iv])+" "+str((coll+nocoll)*0.5)+"\n")
                    #bcr=(np.min(nocoll))*0.5
