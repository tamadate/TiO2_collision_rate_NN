import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim
import glob
import numpy as np
import TiO2

## --------------   From MD simulation (input data)   -------------- ##

trainInput=np.zeros((0,4))  # [n1,n2,b,v]
labelInput=np.zeros(0)

for i in np.arange(3):
    MDdata=TiO2.TiO2()
    MDdata.sizeSet(i)
    MDdata.tempSet(1000)
    MDdata.mapping()
    j=0
    for pss in MDdata.pArray:
        for ib in np.arange(np.size(MDdata.bs)):
            for iv in np.arange(np.size(MDdata.vs)):
                addData=np.array([[MDdata.ns[MDdata.n1],MDdata.ns[j],MDdata.bs[ib],MDdata.vs[iv]]])
                trainInput=np.append(trainInput,addData,axis=0)
                labelInput=np.append(labelInput,pss[iv][ib])
        j+=1


## --------------   From MD simulation (input data)   -------------- ##

train_data, test_data, train_label, test_label = train_test_split(trainInput, labelInput, test_size=0.2)
print("train_data size: {}".format(len(train_data)))
print("test_data size: {}".format(len(test_data)))
print("train_label size: {}".format(len(train_label)))
print("test_label size: {}".format(len(test_label)))

train_x = torch.Tensor(train_data)
test_x = torch.Tensor(test_data)
train_y = torch.LongTensor(train_label)
test_y = torch.LongTensor(test_label)

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

train_batch = DataLoader(
    dataset = train_dataset,
    batch_size = 5,# batch size
    shuffle = True,
    num_workers = 2
)

test_batch = DataLoader(
    dataset = test_dataset,
    batch_size = 5,
    shuffle = False,
    num_workers = 2
)


class Net(nn.Module):
  def __init__(self, D_in, H, D_out):
    super(Net, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H)
    self.linear2 = torch.nn.Linear(H, H2)
    self.linear3 = torch.nn.Linear(H2, H3)
    self.linear4 = torch.nn.Linear(H3, D_out)
  def forward(self, x):
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = F.relu(self.linear3(x))
    x = self.linear4(x)
    return x

D_in = 4	#input dimension
H = 10	#1st hidden layer dimension
H2 = 5	#2nd hidden layer dimension
H3 = 5	#3rd hidden layer dimension
D_out = 2	#output dimension
epoch = 200	#number of training

##device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

net = Net(D_in, H, D_out).to(device)
print("Device: {}".format(device))

#loss function
criterion = nn.CrossEntropyLoss()
#optimize function
optimizer = optim.Adam(net.parameters())

train_loss_list = []#training losses
train_accuracy_list = []#accuracy rate
test_loss_list = []#evaluation losses
test_accuracy_list = []#accuracy rate of test

#main run
for I in range(epoch):
    print('--------')
    print("Epoch: {}/{}".format(I + 1, epoch))
    #initialize training and loss
    train_loss = 0
    train_accuracy = 0
    train_correct = 0
    test_loss = 0
    test_accuracy = 0
    test_correct = 0

# Training mode
    net.train()
    # Main training
    for data, label in train_batch:
        data=data.to(device)
        label=label.to(device)
        optimizer.zero_grad()   # Initialize gradient
        y_pred_prob=net(data) # Calculate predict value
        loss=criterion(y_pred_prob, label)    # Calculate loss
        loss.backward()         # Calculate gradient
        optimizer.step()        # Update parameters
        train_loss+=loss.item()   # Accumulate losses
        y_pred_label=torch.max(y_pred_prob, 1)[1] # Calculate predicted label from predict probability
        train_accuracy+=torch.sum(y_pred_label==label).item()/len(label) # Conunt correct labels

    # Calculate average losses & accuracy in each batch
    batch_train_loss=train_loss/len(train_batch)
    batch_train_accuracy=train_accuracy/len(train_batch)

# Evaluation mode
    net.eval()
    with torch.no_grad(): # Auto grad = 0
        for data, label in test_batch:
            data=data.to(device)
            label=label.to(device)
            y_pred_prob=net(data) # Calculate predict value
            loss=criterion(y_pred_prob, label)    # Calculate loss
            test_loss+=loss.item()    # Accumulate losses
            y_pred_label=torch.max(y_pred_prob, 1)[1] # Calculate predicted label from predict probability
            test_accuracy+=torch.sum(y_pred_label==label).item()/len(label)   # Conunt correct labels

    # Calculate average losses & accuracy in each batch
    batch_test_loss=test_loss/len(test_batch)
    batch_test_accuracy=test_accuracy/len(test_batch)

    # Display losses & accuracy
    print("Train_Loss: {:.4f} Train_Accuracy: {:.4f}".format(batch_train_loss, batch_train_accuracy))
    print("Test_Loss: {:.4f} Test_Accuracy: {:.4f}".format(batch_test_loss, batch_test_accuracy))

    # Make losses & accuracy lists
    train_loss_list.append(batch_train_loss)
    train_accuracy_list.append(batch_train_accuracy)
    test_loss_list.append(batch_test_loss)
    test_accuracy_list.append(batch_test_accuracy)


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
fig, axs = plt.subplots(1,2,figsize=(11,5))
for i in np.arange(2):
	axNormal(axs.flat[i])

## Plotting results
axs.flat[0].set_title('(a) Train & Test losses',loc="left",size=15)
axs.flat[0].set_ylabel("Loss [-]",size=15)
axs.flat[0].set_xlabel('Epoch [-]',size=15)
axs.flat[0].plot(range(1, epoch+1), train_loss_list, color='blue',linestyle='-', label="Train ($N$ = "+str(len(train_label))+")")
axs.flat[0].plot(range(1, epoch+1), test_loss_list, color='red', linestyle='--', label="Test ($N$ = "+str(len(test_label))+")")
axs.flat[0].legend(frameon=False)

axs.flat[1].set_title('(b) Train & Test accuracies',loc="left",size=15)
axs.flat[1].set_ylabel("Accuracy [-]",size=15)
axs.flat[1].set_xlabel('Epoch [-]',size=15)
axs.flat[1].plot(range(1, epoch+1), train_accuracy_list, color='blue',linestyle='-', label="Train ($N$ = "+str(len(train_label))+")")
axs.flat[1].plot(range(1, epoch+1), test_accuracy_list, color='red', linestyle='--', label="Test ($N$ = "+str(len(test_label))+")")
axs.flat[1].legend(frameon=False)

plt.savefig("NNresult.png", dpi=1000)
plt.show()

torch.save(net.state_dict(),"../NN_learned/TiO2_learned.pth")
