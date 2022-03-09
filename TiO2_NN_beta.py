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
    vmax=200
    vmin=1
    bmax=200
    bmin=0
    Nb=100
    Nv=100

    db=(bmax-bmin)/Nb
    dv=(vmax-vmin)/Nv
        
    def calculatePNN(self,n1,n2,b,v):
        calTensor=torch.tensor([[float(n1),float(n2),float(b),float(v)]])
        return torch.max(TiO2.model(calTensor),1)[1].tolist()

    def calculateBetaNN(self,n1,n2,T):
        kbT=TiO2.kb*T
        n=np.array((n1,n2))
        m=n/3.0*(48+32)/1000.0/6.02e23
        mij=1/(1/m[0]+1/m[1])
        coeff=(mij*0.5/np.pi/kbT)**1.5*8*np.pi**2
        beta=0
        for ib in np.arange(TiO2.Nb):
            b=(TiO2.bmin+TiO2.db*ib)
            for iv in np.arange(TiO2.Nv):
                v=TiO2.vmin+TiO2.dv*iv
                calTensor=torch.tensor([[float(n1),float(n2),float(b),float(v)]])
                predictLabel=torch.max(TiO2.model(calTensor),1)[1].tolist()
                p=predictLabel[0]
                length=coeff*v**3*np.exp(-mij*v**2*0.5/kbT)*b*p
                beta+=length*TiO2.db*TiO2.dv
        return beta*1e-20

    def NtoD(self,N):
        return 10.0*(N/48.0)**0.33333

    def DtoN(self,D):
        return (D/10.0)**3*48.0




