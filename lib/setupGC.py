import random
from random import choices
import numpy as np
import pandas as pd
import os

import torch
from torch_geometric.transforms import OneHotDegree

# from models import GIN, serverGIN, GIN_dc, serverGIN_dc, GCN_dc, serverGCN_dc, GCN
from models import *
from server import *
from client import *
from utils import *

from scipy.special import rel_entr
import scipy
from torch_geometric.utils import erdos_renyi_graph, degree


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
def _randChunk(graphs, num_client, overlap, seed=None): ## 随机分配图数据集，分给各个client
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum/num_client))
    graphs_chunks = []
    if not overlap:
        for i in range(num_client):
            graphs_chunks.append(graphs[i*minSize:(i+1)*minSize])
        for g in graphs[num_client*minSize:]:
            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_client)
        for s in sizes:
            graphs_chunks.append(choices(graphs, k=s))
    return graphs_chunks

def js_diver(P,Q):
    M=P+Q
    return 0.5*scipy.stats.entropy(P,M,base=2)+0.5*scipy.stats.entropy(Q,M,base=2)



def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)    
    return torch.load(fpath, map_location=torch.device('cpu'))

def get_data(args, client_id):
    return [
        torch_load(
            "", 
            f'{args.dataset}_{args.mode}/{args.clients}/partition_{client_id}.pt'
        )['client_data']
    ]
    
def get_data_init(args, client_id):
    if args.alg == "fedstar":
        return [
        torch_load(
            "",
            f'{args.dataset}_{args.mode}/{args.clients}/initfedstar_{client_id}.pt'
        )['client_data']
    ]
    elif args.alg == "fedcap":
        return [
        torch_load(
            "/home/1005wjy/datasets/",
            f'{args.dataset}_{args.mode}/{args.clients}/init_{client_id}.pt'
        )['client_data']
    ]
    else:
        return [
            torch_load(
                "/home/1005wjy/datasets/", 
                f'{args.dataset}_{args.mode}/{args.clients}/partition_{client_id}.pt'
            )['client_data']
        ]        
    
    
def get_homogeneous_edge(g):
    edge = []
    b, e = g.edge_index[0], g.edge_index[1]
    for i in range(len(b)):
        if g.y[b[i]]==g.y[e[i]]:
            edge.append([int(b[i]),int(e[i])])
    return edge
    
def prepareData_nodeTask(args):
    ## load dataset
    splitedData = {}

    if args.dataset == "Cora":
        num_features = 1433
        num_labels = 7
    if args.dataset == "CiteSeer":
        num_features = 3703
        num_labels = 6
    if args.dataset == "PubMed":
        num_features = 500
        num_labels = 3
    if args.dataset == 'Computers':
        num_features = 767
        num_labels = 10       
    if args.dataset == 'Photo':
        num_features = 745
        num_labels = 8      
    if args.dataset == "Reddit":
        num_features = 602
        num_labels = 41   

    if args.dataset == 'Ogbn':
        num_features = 128
        num_labels = 40

    if args.dataset == 'CS':
        num_features = 6805
        num_labels = 15    
        
    if args.dataset == 'Physics':
        num_features = 8415
        num_labels = 5            

    if args.dataset == 'CoraFull':
        num_features = 8710
        num_labels = 70  

    if args.dataset == 'Reddit2':
        num_features = 602
        num_labels = 41

    if args.dataset == 'Products':
        num_features = 100
        num_labels =  47

    if args.dataset == 'Flickr':
        num_features = 500
        num_labels =  7

    if args.dataset == 'Yelp':
        num_features = 300
        num_labels =  100
        
    # homogeneous_edge = []
    for c in range(args.clients):
        client_graph = get_data_init(args, c)[0]
        train_size = len(client_graph.y[client_graph.train_mask])
        print(train_size)
        
        # if args.alg == 'fedstar':
        #     client_graph = init_structure_encoding_node(args, client_graph)     

        splitedData[c] = (client_graph, num_features, num_labels, train_size)
        
    # return splitedData, df, homogeneous_edge
    return splitedData

def setup_devices_nodeTask(splitedData, args):
    clients = []
    for c in range(args.clients):
        client_graph, num_features, num_labels, train_size = splitedData[c]
        if args.alg == 'fedstar':
            cmodel_gc = GCN_dc(num_features, args.hidden, num_labels, args.dropout, args.n_se)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            # set_seed(args)
            clients.append(   Client_GC(cmodel_gc, c, train_size, client_graph, optimizer, args)   )     
        elif args.alg == 'fedcap':
            main_model = GCN(num_features, args.hidden, num_labels, args.dropout) 
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, main_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            set_seed(args)
            
            struct_model = GCN_str(num_features, args.hidden, args.hidden_str, num_labels, args.dropout, args.n_rw)            
            optimizer_struct = torch.optim.Adam(filter(lambda p: p.requires_grad, struct_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            set_seed(args)
            
            clients.append(   Client_gr(main_model, c, train_size, client_graph, optimizer, args, struct_model, optimizer_struct, num_labels)   )     
            
        else: ## fedavg and fedprox and gcfl plus
            cmodel_gc = GCN(num_features, args.hidden, num_labels, args.dropout)       
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            # set_seed(args)
            clients.append(   Client_GC(cmodel_gc, c, train_size, client_graph, optimizer, args)   )
   
        
    if args.alg == 'fedstar':
        smodel = GCN_dc(num_features, args.hidden, num_labels, args.dropout, args.n_se)
        server = Server(smodel, args.device)
        # set_seed(args)
    elif args.alg == 'fedcap':
        main_model = GCN(num_features, args.hidden, num_labels, args.dropout) 
        set_seed(args)
        
        str_model = GCN_str(num_features, args.hidden, args.hidden_str, num_labels, args.dropout, args.n_rw)  
        set_seed(args)
        total_params_gcn = sum(p.numel() for p in main_model.parameters() if p.requires_grad)
        total_params_str = sum(p.numel() for p in str_model.parameters() if p.requires_grad)
        print("model param.:",total_params_gcn*4/float(1024),total_params_str*4/float(1024),"KB")
        all_model = []
        for i in range(args.clients):
            m_model = GCN(num_features, args.hidden, num_labels, args.dropout) 
            all_model.append(m_model)
            set_seed(args)
        server = Server_gr(main_model, args.device, str_model, num_features, num_labels, args, all_model)
        

        
    else: ## fedavg and fedprox and gcfl plus
        smodel = GCN(num_features, args.hidden, num_labels, args.dropout) 
        server = Server(smodel, args.device)
    
    
    return clients, server