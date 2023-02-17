import torch
from torchvision import datasets
from torch import nn, optim, autograd
import torchvision
from backpack import backpack, extend
from backpack.extensions import BatchGrad


class Net(nn.Module):
    def __init__(self, mlp,topmlp):
        super(Net, self).__init__()
        self.net = nn.Sequential(mlp,topmlp)
    def forward(self,data):
        return self.net(data)
# Define and instantiate the model
class Linear(nn.Module):
    def __init__(self, hidden_dim=1, input_dim=2*14*14):
        super(Linear, self).__init__()

        self.input_dim = input_dim

        lin1 = nn.Linear(self.input_dim, hidden_dim)
        
        nn.init.xavier_uniform_(lin1.weight)
        nn.init.zeros_(lin1.bias)

        self._main = lin1 
    def forward(self,input):
        out = input.view(input.shape[0], self.input_dim)
        out = self._main(out)
        return out


class MLP(nn.Module):
    def __init__(self, hidden_dim=390, input_dim=2*14*14):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim

        lin1 = nn.Linear(self.input_dim, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)

        nn.init.xavier_uniform_(lin1.weight)
        nn.init.zeros_(lin1.bias)
        nn.init.xavier_uniform_(lin2.weight)
        nn.init.zeros_(lin2.bias)

        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True))
        
    def forward(self, input):
        out = input.view(input.shape[0], self.input_dim)
        out = self._main(out)
        return out


class TopMLP(nn.Module):
    def __init__(self, hidden_dim=390, n_top_layers=1, n_targets=1, fishr=False):

        super(TopMLP, self).__init__()

        if fishr:
            self.lin1 = lin1 = extend(nn.Linear(hidden_dim,n_targets))
        else:
            self.lin1 = lin1 = nn.Linear(hidden_dim,n_targets)
        nn.init.xavier_uniform_(lin1.weight)
        nn.init.zeros_(lin1.bias)
        self._main = nn.Sequential(lin1)
        self.weights = [lin1.weight, lin1.bias]

    def forward(self,input):
        out = self._main(input)
        return out 



# # from https://github.com/facebookresearch/DomainBed/tree/master/domainbed
class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
