import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
  def __init__(self, D_in, H, D_out):
    super(Net, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H[0])
    self.linear2 = torch.nn.Linear(H[0], H[1])
    self.linear3 = torch.nn.Linear(H[1], H[2])
    self.linear4 = torch.nn.Linear(H[2], D_out)
  def forward(self, x):
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = F.relu(self.linear3(x))
    x = self.linear4(x)
    return x


class TiO2:
    model_path = 'TiO2_learned.pth'
    D_in = 4	#input dimension
    H = [10,5,5]	#hidden layer dimensions
    D_out = 2	#output dimension
    epoch = 20	#number of training
    device = torch.device('cpu')
    model = Net(D_in, H, D_out).to(device)
    model.load_state_dict(torch.load(model_path))
    kb=1.38e-23
    R=8.314
    Mmono=(16+16+48)*1e-3
    vmax=1000
    vmin=1
    bmax=200
    bmin=0
    Nb=100
    Nv=100
    mij=-1
    db=(bmax-bmin)/Nb
    dv=(vmax-vmin)/Nv
    dbdv=db*dv
    barray=np.arange(bmin,bmin+db*Nb,db)
    varray=np.arange(vmin,vmin+dv*Nb,dv)
    ps=np.zeros((Nb,Nv))

    def calculatePNN(self,n1,n2,b,v):
        calTensor=torch.tensor([[float(n1),float(n2),float(b),float(v)]])
        return torch.max(TiO2.model(calTensor),1)[1].tolist()

    def calculateBetaNN(self,n1,n2,T):
        n=np.array((n1,n2))
        m=n/3.0*self.Mmono				# Molar mass of clusters [kg/mol]
        self.mij=m[0]*m[1]/(m[0]+m[1])		# Reduced molar mass [kg/mol]
        mijxtwoRT_inv=self.mij/(self.R*T*2.0)# coefficient
        coeff=(self.mij*0.5/np.pi/T/self.R)**1.5*8*np.pi*np.pi*2**0.5
        beta=0
        ib=0
        for b in self.barray:
            iv=0
            for v in self.varray:
                calTensor=torch.tensor([[float(n1),float(n2),float(b),float(v)]])
                predictLabel=torch.max(self.model(calTensor),1)[1].tolist()
                self.ps[iv][ib]=predictLabel[0]
                length=coeff*v**3*np.exp(-mijxtwoRT_inv*v*v)*b*self.ps[iv][ib]
                beta+=length*self.dbdv
                iv+=1
            ib+=1
        return beta*1e-20


    def NtoD(self,N):
        return 10.0*(N/48.0)**0.33333

    def DtoN(self,D):
        return (D/10.0)**3*48.0
