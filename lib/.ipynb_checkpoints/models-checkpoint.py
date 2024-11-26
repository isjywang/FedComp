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
        # self.hid = self.nrw + 2
        self.output_hidden = max(1,int(self.input_hidden/self.nhid*self.nclass))
        self.structural_layer_1 = GCNConv(self.input_hidden, self.nhid_str)  ## RandomWalk + Homogeneity
        self.structural_layer_2 = GCNConv(self.nhid_str, self.output_hidden)        
        # self.structural_layer_1 = SAGEConv(self.input_hidden, self.nhid_str, aggr='mean')
        # self.structural_layer_2 = SAGEConv(self.nhid_str, self.output_hidden, aggr='mean')
        # self.conv1 = GATConv(self.input_hidden, self.nhid_str, heads=8, concat=True)
        # self.conv2 = GATConv(self.nhid_str * 8, self.output_hidden, heads=1, concat=False)  
        self.clsif = torch.nn.Linear(self.nclass+ self.output_hidden, self.nclass)
        # print("str model output hidden:",self.output_hidden)

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







# class SAGE(torch.nn.Module):
#     def __init__(self, n_feat, nhid, nclass, dropout): ## n_se 1
#         super(SAGE,self).__init__()
#         from torch_geometric.nn import SAGEConv

#         self.dropout = dropout
#         self.n_feat = n_feat
#         self.nhid = nhid
#         self.nclass = nclass

#         self.conv1 = SAGEConv(self.n_feat, self.nhid, aggr='mean')
#         self.conv2 = SAGEConv(self.nhid, self.nhid, aggr='mean')
#         self.clsif = torch.nn.Linear(self.nhid, self.nclass)
        
#     def forward(self, data, gr=False):
#         x, edge_index = data.x, data.edge_index 

#         x = self.conv1(x, edge_index) ## x,hid
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)

#         x1 = self.conv2(x, edge_index)
#         x2 = F.relu(x1)
#         x3 = F.dropout(x2, training=self.training)
#         x = self.clsif(x3)
#         return x

    
#     def loss(self, pred, label):
#         return F.cross_entropy(pred, label)



# class GAT(torch.nn.Module):
#     def __init__(self, n_feat, nhid, nclass, dropout, num_heads=8): 
#         super(GAT,self).__init__()
#         from torch_geometric.nn import GATConv

#         self.dropout = dropout
#         self.n_feat = n_feat
#         self.nhid = nhid
#         self.nclass = nclass
        
#         self.conv1 = GATConv(self.n_feat, self.nhid, heads=num_heads, concat=True)
#         self.conv2 = GATConv(self.nhid * num_heads, self.nhid, heads=1, concat=False)  
#         self.clsif = torch.nn.Linear(self.nhid, self.nclass)
        
#     def forward(self, data, gr=False):
#         x, edge_index = data.x, data.edge_index 

#         x = self.conv1(x, edge_index) ## x,hid
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)

#         x1 = self.conv2(x, edge_index)
#         x2 = F.relu(x1)
#         x3 = F.dropout(x2, training=self.training)
#         x = self.clsif(x3)
#         return x

    
#     def loss(self, pred, label):
#         return F.cross_entropy(pred, label)



# class Concat_model(torch.nn.Module):   ## Concat Emb_gnn and Emb_str by a Linear, then classification.
#     def __init__(self, nhid, nhidstr, nclass):
#         super(Concat_model,self).__init__()
#         self.nhid, self.nhidstr, self.nclass = nhid, nhidstr, nclass
#         self.clsif = torch.nn.Linear(self.nhid+self.nhidstr, self.nclass)

#     def forward(self, inputf, repair=False):
#         x = self.clsif(inputf)
#         results = F.log_softmax(x, dim=1)
#         if repair:
#             return x, results
#         else:
#             return x
    
    # def loss(self, pred, label):
    #     return F.nll_loss(pred, label)







# ## 方案一：feature先过GCN 1 , 再来concat (还没尝试)
# ## 方案二：feature直接concat (now)
# ## 方案三: 两个都先过一层，再concat (还没尝试)

# class GCN_gr(torch.nn.Module):
#     def __init__(self, n_feat, nhid, nclass, nlayer, dropout, n_rw): ## n_se 1
#         super(GCN_gr,self).__init__()
#         from torch_geometric.nn import GCNConv
#         self.dropout, self.n_feat, self.nhid, self.nclass, self.nrw = dropout, n_feat, nhid, nclass, n_rw

#         self.structual_layer_1 = GCNConv(self.nrw + self.nclass + self.n_feat, self.nhid)  ## RandomWalk + Homogeneity
#         # self.structual_layer_1 = GCNConv(self.nrw + self.n_feat, self.nhid)  ## only RandomWalk Embedding
#         # self.structual_layer_1 = GCNConv(self.nclass + self.n_feat, self.nhid)  ## only Homogeneity Embedding
        
#         self.structual_layer_2 = GCNConv(self.nhid, self.nhid)        
#         self.clsif = torch.nn.Linear(self.nhid, self.nclass) ##server none 

#     def forward(self, data, repair=False):
#         x, edge_index, structual_emb = data.x, data.edge_index, data.stc_enc 
#         # x = self.feature_hid(x)
#         x = torch.cat((x, structual_emb), -1)  
#         x = self.structual_layer_1(x, edge_index) ## x,hid
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)

#         x = self.structual_layer_2(x, edge_index)
#         if repair:
#             return x
#         else:
#             x = F.relu(x)
#             x = F.dropout(x, training=self.training)   
            
#             x = self.clsif(x)
#             x = F.log_softmax(x, dim=1)
#             return x
    
#     def loss(self, pred, label):
#         return F.nll_loss(pred, label)























# class serverGIN(torch.nn.Module):
#     def __init__(self, nlayer, nhid):
#         super(serverGIN, self).__init__()
#         self.graph_convs = torch.nn.ModuleList()
#         self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
#                                        torch.nn.Linear(nhid, nhid))
#         self.graph_convs.append(GINConv(self.nn1))
#         for l in range(nlayer - 1):
#             self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
#             self.graph_convs.append(GINConv(self.nnk))

# class GIN(torch.nn.Module):
#     def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
#         super(GIN, self).__init__()
#         self.num_layers = nlayer
#         self.dropout = dropout

#         self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

#         self.graph_convs = torch.nn.ModuleList()
#         self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
#         self.graph_convs.append(GINConv(self.nn1))
#         for l in range(nlayer - 1):
#             self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
#             self.graph_convs.append(GINConv(self.nnk))

#         self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
#         self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.pre(x)
#         for i in range(len(self.graph_convs)):
#             x = self.graph_convs[i](x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, self.dropout, training=self.training)
#         x = global_add_pool(x, batch)
#         x = self.post(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.readout(x)
#         x = F.log_softmax(x, dim=1)
#         return x

#     def loss(self, pred, label):
#         return F.nll_loss(pred, label)


# class serverGIN_dc(torch.nn.Module):
#     def __init__(self, n_se, nlayer, nhid):
#         super(serverGIN_dc, self).__init__()

#         self.embedding_s = torch.nn.Linear(n_se, nhid)
#         self.Whp = torch.nn.Linear(nhid + nhid, nhid)

#         self.graph_convs = torch.nn.ModuleList()
#         self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid + nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
#         self.graph_convs.append(GINConv(self.nn1))
#         self.graph_convs_s_gcn = torch.nn.ModuleList()
#         self.graph_convs_s_gcn.append(GCNConv(nhid, nhid))

#         for l in range(nlayer - 1):
#             self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid + nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
#             self.graph_convs.append(GINConv(self.nnk))
#             self.graph_convs_s_gcn.append(GCNConv(nhid, nhid))

# class GIN_dc(torch.nn.Module):
#     def __init__(self, nfeat, n_se, nhid, nclass, nlayer, dropout): ## n_se 1
#         super(GIN_dc, self).__init__()
#         self.num_layers = nlayer
#         self.dropout = dropout

#         self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))##server none

#         self.embedding_s = torch.nn.Linear(n_se, nhid) ## 2

#         self.graph_convs = torch.nn.ModuleList()
#         self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid + nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))      ## 3
        
#         self.graph_convs.append(GINConv(self.nn1))
        
#         self.graph_convs_s_gcn = torch.nn.ModuleList()  ## 4
#         self.graph_convs_s_gcn.append(GCNConv(nhid, nhid)) ## 5

#         for l in range(nlayer - 1):
#             self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid + nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid)) ## 6
#             self.graph_convs.append(GINConv(self.nnk))
#             self.graph_convs_s_gcn.append(GCNConv(nhid, nhid)) ## 7

#         self.Whp = torch.nn.Linear(nhid + nhid, nhid) ## 8
#         self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU()) ##server none
#         self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))##server none

#     def forward(self, data):
#         x, edge_index, batch, s = data.x, data.edge_index, data.batch, data.stc_enc  ## 多了s 9
#         x = self.pre(x) ##x,hid
#         s = self.embedding_s(s) ## 10  x,hid
#         for i in range(len(self.graph_convs)):
#             x = torch.cat((x, s), -1)  ## 11 x,2hid
#             x = self.graph_convs[i](x, edge_index) ## x hid
#             x = F.relu(x)
#             x = F.dropout(x, self.dropout, training=self.training)
#             s = self.graph_convs_s_gcn[i](s, edge_index) ## 12 x,hid
#             s = torch.tanh(s) ## 13

#         x = self.Whp(torch.cat((x, s), -1)) ##13  2x,hid --> x,hid
        
#         x = global_add_pool(x, batch)
#         x = self.post(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.readout(x)
#         x = F.log_softmax(x, dim=1)
#         return x

#     def loss(self, pred, label):
#         return F.nll_loss(pred, label)


# class serverGraphSage(torch.nn.Module):
#     def __init__(self, nlayer, nhid):
#         super(serverGraphSage, self).__init__()
#         self.graph_convs = torch.nn.ModuleList()
#         self.graph_convs.append(SAGEConv(nhid, nhid))
#         for l in range(nlayer - 1):
#             self.graph_convs.append(SAGEConv(nhid, nhid))

# class GraphSage(torch.nn.Module):
#     def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
#         super(GraphSage, self).__init__()
#         self.num_layers = nlayer
#         self.dropout = dropout

#         self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

#         self.graph_convs = torch.nn.ModuleList()
#         self.graph_convs.append(SAGEConv(nhid, nhid))

#         for l in range(nlayer - 1):
#             self.graph_convs.append(SAGEConv(nhid, nhid))

#         self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
#         self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.pre(x)
#         for i in range(len(self.graph_convs)):
#             x = self.graph_convs[i](x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, self.dropout, training=self.training)
#         x = global_add_pool(x, batch)
#         x = self.post(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.readout(x)
#         x = F.log_softmax(x, dim=1)
#         return x

#     def loss(self, pred, label):
#         return F.nll_loss(pred, label)
    
    

    
    
    
# class serverGCN_dc(torch.nn.Module):
#     def __init__(self, n_feat, n_se, nhid, nclass, nlayer, dropout): ## n_se 1
#         super(serverGCN_dc,self).__init__()
#         from torch_geometric.nn import GCNConv
#         self.dropout = dropout
#         self.n_feat = n_feat
#         self.nhid = nhid
#         self.nclass = nclass
        
#         self.embedding_s = torch.nn.Linear(n_se, nhid) ## 2
        
#         # self.feature_hid = torch.nn.Linear(self.n_feat, self.nhid) ##server none
#         self.conv1 = GCNConv(self.nhid+self.nhid, self.nhid)
#         self.conv2 = GCNConv(self.nhid+self.nhid, self.nhid)
#         self.clsif = torch.nn.Linear(self.nhid, self.nclass) ##server none 
        
#         self.graph_conv_s1 = GCNConv(self.nhid, self.nhid)
#         self.graph_conv_s2 = GCNConv(self.nhid, self.nhid)
#         self.Whp = torch.nn.Linear(self.nhid + self.nhid, self.nhid) ## 8

