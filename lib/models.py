import torch
import torch.nn.functional as F
import random
from torch_geometric.nn import *
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class GCN_dc(torch.nn.Module):
    def __init__(self, n_feat, nhid, nclass, dropout, n_se): 
        super(GCN_dc,self).__init__()
        
        self.dropout = dropout
        self.n_feat = n_feat
        self.nhid = nhid
        self.nclass = nclass
        
        self.embedding = torch.nn.Linear(n_se, nhid) ## 2
        
        # self.feature_hid = torch.nn.Linear(self.n_feat, self.nhid) ##server none
        self.conv1 = GCNConv(self.n_feat+self.nhid, self.nhid)
        self.conv2 = GCNConv(self.nhid+self.nhid, self.nhid)
        self.clsif = torch.nn.Linear(self.nhid, self.nclass) ##server none 
        
        self.graph_conv_s1 = GCNConv(self.nhid, self.nhid)
        self.graph_conv_s2 = GCNConv(self.nhid, self.nhid)
        self.Whp = torch.nn.Linear(self.nhid + self.nhid, self.nhid) ## 8

    def forward(self, data):
        x, edge_index, batch, s = data.x, data.edge_index, data.batch, data.stc_enc  ## 多了s 9
        # x = self.feature_hid(x)
        s = self.embedding(s) ## 10  x,hid
        
        x = torch.cat((x, s), -1)  ## 11 x,hid+hid
        x = self.conv1(x, edge_index) ## x,hid
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        s = self.graph_conv_s1(s, edge_index) ## 12 x,hid
        s = torch.tanh(s) ## 13
        
        x = torch.cat((x, s), -1)  ## 11 x,hid+hid
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        s = self.graph_conv_s2(s, edge_index) ## 12 x,hid
        s = torch.tanh(s) ## 13        
        
        x = self.Whp(torch.cat((x, s), -1)) ##13  x,2hid --> x,hid
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.clsif(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def loss(self, pred, label):
        return F.nll_loss(pred, label)

        

class GCN(torch.nn.Module):
    def __init__(self, n_feat, nhid, nclass, dropout): ## n_se 1
        super(GCN,self).__init__()
        from torch_geometric.nn import GCNConv

        self.dropout = dropout
        self.n_feat = n_feat
        self.nhid = nhid
        self.nclass = nclass
        # self.conv1 = SAGEConv(self.n_feat, self.nhid, aggr='mean')
        # self.conv2 = SAGEConv(self.nhid, self.nhid, aggr='mean')
        self.conv1 = GCNConv(self.n_feat, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        # self.conv1 = GATConv(self.n_feat, self.nhid, heads=8, concat=True)
        # self.conv2 = GATConv(self.nhid * 8, self.nhid, heads=1, concat=False)          
        self.clsif = torch.nn.Linear(self.nhid, self.nclass)
        self.nhid = nhid
        
    def forward(self, data, gr=False):
        x, edge_index = data.x, data.edge_index 

        x = self.conv1(x, edge_index) ## x,hid
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x1 = self.conv2(x, edge_index)
        x2 = F.relu(x1)
        x3 = F.dropout(x2, training=self.training)
        x = self.clsif(x3)
        return x
        # if gr:
        #     return x2, x
        # else:
        # # x = F.log_softmax(x, dim=1)
        #     return x
    
    def loss(self, pred, label):
        return F.cross_entropy(pred, label)
        # return F.nll_loss(pred, label)

class GCN_str(torch.nn.Module):  ## RandomWalk Embedding + Homogeneity Embedding
    def __init__(self, n_feat, nhid, nhid_str, nclass, dropout, n_rw):
        super(GCN_str,self).__init__()
        self.dropout, self.n_feat, self.nhid, self.nhid_str, self.nclass, self.nrw = dropout, n_feat, nhid, nhid_str, nclass, n_rw
        
        self.input_hidden = self.nrw + 2
        self.output_hidden = max(1,int(self.input_hidden/self.nhid*self.nclass))
        self.structural_layer_1 = GCNConv(self.input_hidden, self.nhid_str)  ## RandomWalk + Homogeneity
        self.structural_layer_2 = GCNConv(self.nhid_str, self.output_hidden)        
        self.clsif = torch.nn.Linear(self.nclass+ self.output_hidden, self.nclass)


    def forward(self, data, emb_a):
        x, edge_index, structual_emb = data.x, data.edge_index, data.stc_enc
        x = self.structural_layer_1(data.stc_enc, edge_index)

        x = F.dropout(x, training=self.training)

        x = self.structural_layer_2(x, edge_index)
        x = F.relu(x)
        emb_b = F.dropout(x, training=self.training)

        emb = torch.cat((emb_a, emb_b), dim=-1)
        x = self.clsif(emb)
        
        return x
        
    def loss(self, pred, label):
        return F.cross_entropy(pred, label)















