import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
# from torch_geometric.datasets import QM9
from torch_geometric.nn import GCNConv,global_add_pool,ChebConv,global_max_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Dataset, Data, DataLoader


from infomax import *

class JavaEncoder(torch.nn.Module):
    def __init__(self, num_features, dim):
        super(JavaEncoder, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)
        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = ChebConv(dim, dim, K=2,normalization='sym')
        
        # self.gru = GRU(dim, dim)
        # self.set2set = Set2Set(dim, processing_steps=3)

    def forward(self, data,batch):
        out = F.relu(self.lin0(data.x2.float()))
        feat_map = []
        # for i in range(3):
        # out = F.relu(self.conv(out, data.edge_index2.long()))
        # out, h = self.gru(m.unsqueeze(0), h)
        # out = out.squeeze(0)
        # print(out.shape) : [num_node x dim]

        # out = Node2Vec(data.edge_index2.long(), embedding_dim=128, walk_length=20,context_size=10, walks_per_node=10,num_negative_samples=1, p=1, q=1, sparse=True)
       
        out = F.relu(self.conv(out, data.edge_index2.long()))
        feat_map.append(out)

        out = global_add_pool(out, batch)
        # print(out.shape)

        return out, feat_map[-1]

class PyEncoder(torch.nn.Module):
    def __init__(self, num_features, dim):
        super(PyEncoder, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)
        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = ChebConv(dim, dim, K=2,normalization='sym')

        # self.gru = GRU(dim, dim)
        # self.set2set = Set2Set(dim, processing_steps=3)

    def forward(self, data,batch):
        out = F.relu(self.lin0(data.x1.float()))
        feat_map = []
        # for i in range(3):
        # out = F.relu(self.conv(out, data.edge_index1.long()))
        # out, h = self.gru(m.unsqueeze(0), h)
        # out = out.squeeze(0)
        # print(out.shape) : [num_node x dim]

        # out = Node2Vec(data.edge_index1.long(), embedding_dim=128, walk_length=20,context_size=10, walks_per_node=10,num_negative_samples=1, p=1, q=1, sparse=True)
        out = F.relu(self.conv(out, data.edge_index1.long()))
        feat_map.append(out)
        out = global_add_pool(out, batch)
        # print(out.shape)

        return out, feat_map[-1]

##### REMOVED FROM THE MAIN SCRIPT
# class PriorDiscriminator(nn.Module):
    # def __init__(self, input_dim):
        # super().__init__()
        # self.l0 = nn.Linear(input_dim, input_dim)
        # self.l1 = nn.Linear(input_dim, input_dim)
        # self.l2 = nn.Linear(input_dim, 1)

    # def forward(self, x):
        # h = F.relu(self.l0(x))
        # h = F.relu(self.l1(h))
        # return torch.sigmoid(self.l2(h))

class FF(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class Net(torch.nn.Module):
    def __init__(self, num_features, dim, use_unsup_loss=False, separate_encoder=False):
        super(Net, self).__init__()

        self.embedding_dim = dim
        self.separate_encoder = separate_encoder
        # self.local = True
        # self.prior = False

        # java encoder
        self.encoder2 = JavaEncoder(num_features, dim)
        
        # python encoder
        self.encoder1 = PyEncoder(num_features, dim)

        # FCs after concat
        # self.fc1 = torch.nn.Linear(2 * dim, dim)
        # self.fc2 = torch.nn.Linear(dim, 1)

        self.fc2 = torch.nn.Linear(2*dim, 1)
       
        self.init_emb()

    def init_emb(self):
      initrange = -1.5 / self.embedding_dim
      for m in self.modules():
          if isinstance(m, nn.Linear):
              torch.nn.init.xavier_uniform_(m.weight.data)
              if m.bias is not None:
                  m.bias.data.fill_(0.0)


    def forward(self, data):
        # print(data)

        out1, M1 = self.encoder1(data,data.x1_batch)
        out2, M2 = self.encoder2(data, data.x2_batch)

        # print(out1.shape)
        # print(out2.shape)

        concatenatedEmb = torch.cat((out1,out2),dim=1)
        # print(concatenatedEmb.shape)

        # concatenatedEmb = F.relu(self.fc1(concatenatedEmb))

        out = self.fc2(concatenatedEmb)
        pred = out.view(-1)
        return pred

    def unsup_loss1(self, data,batch):
        if self.separate_encoder:
            y, M = self.unsup_encoder1(data,batch)
        else:
            y, M = self.encoder1(data,batch)
        g_enc = self.global_d1(y)
        l_enc = self.local_d1(M)

        measure = 'JSD'
        if self.local:
            loss = local_global_loss_(l_enc, g_enc, data.edge_index1.long(),batch, measure)
        return loss

    def unsup_loss2(self, data,batch):
        if self.separate_encoder:
            y, M = self.unsup_encoder2(data,batch)
        else:
            y, M = self.encoder2(data,batch)
        g_enc = self.global_d2(y)
        l_enc = self.local_d2(M)

        measure = 'JSD'
        if self.local:
            loss = local_global_loss_(l_enc, g_enc, data.edge_index2.long(),batch, measure)
        return loss

    def unsup_sup_loss1(self, data,batch):
        y, M =   self.encoder1(data,batch)
        y_, M_ = self.unsup_encoder1(data,batch)

        g_enc = self.ff11(y)
        g_enc1 = self.ff12(y_)

        measure = 'JSD'
        loss = global_global_loss_(g_enc, g_enc1, data.edge_index1.long(), batch, measure)

        return loss

    def unsup_sup_loss2(self, data,batch):
        y, M =   self.encoder2(data,batch)
        y_, M_ = self.unsup_encoder2(data,batch)

        g_enc = self.ff21(y)
        g_enc1 = self.ff22(y_)

        measure = 'JSD'
        loss = global_global_loss_(g_enc, g_enc1, data.edge_index2.long(),batch, measure)

        return loss
    # def align_unsup_sup_loss(self, data):
        # y, M =   self.encoder(data)
        # y_, M_ = self.unsup_encoder(data)
        
        # return  F.mse_loss(y, y_)
