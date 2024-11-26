import torch
from torch_geometric.utils import to_networkx, degree, to_dense_adj, to_scipy_sparse_matrix
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy import sparse as sp
import dgl
import numpy as np
import networkx as nx


def set_config(args):
    ## Experimental Settings
    args.weight_decay = 1e-8
    args.local_epoch = 4 if args.dataset in ['CoraFull'] else 2
    ## FedCap Settings
    args.repair_fre = 1/5 if args.dataset in ['Ogbn'] else 1/4
    args.k = 3+int(min(args.clients,20)/6) if args.dataset in ['Cora','CiteSeer','CoraFull'] else 8 

def split_data(graphs, train=None, test=None, shuffle=True, seed=None):
    y = torch.cat([graph.y for graph in graphs])
    graphs_tv, graphs_test = train_test_split(graphs, train_size=train, test_size=test, stratify=y, shuffle=shuffle, random_state=seed)
    return graphs_tv, graphs_test


def get_numGraphLabels(dataset):
    s = set()
    for g in dataset:
        s.add(g.y.item())
    return len(s)


def _get_avg_nodes_edges(graphs):
    numNodes = 0.
    numEdges = 0.
    numGraphs = len(graphs)
    for g in graphs:
        numNodes += g.num_nodes
        numEdges += g.num_edges / 2.  # undirected
    return numNodes/numGraphs, numEdges/numGraphs


def get_stats(df, ds, graphs_train, graphs_val=None, graphs_test=None):
    df.loc[ds, "#graphs_train"] = len(graphs_train)
    avgNodes, avgEdges = _get_avg_nodes_edges(graphs_train)
    df.loc[ds, 'avgNodes_train'] = avgNodes
    df.loc[ds, 'avgEdges_train'] = avgEdges

    if graphs_val:
        df.loc[ds, '#graphs_val'] = len(graphs_val)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_val)
        df.loc[ds, 'avgNodes_val'] = avgNodes
        df.loc[ds, 'avgEdges_val'] = avgEdges

    if graphs_test:
        df.loc[ds, '#graphs_test'] = len(graphs_test)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_test)
        df.loc[ds, 'avgNodes_test'] = avgNodes
        df.loc[ds, 'avgEdges_test'] = avgEdges

    return df

def init_structure_encoding(args, gs, type_init):

    if type_init == 'rw':
        for g in gs:
            # Geometric diffusion features with Random Walk
            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

            Dinv=sp.diags(D)
            RW=A*Dinv
            M=RW

            SE_rw=[torch.from_numpy(M.diagonal()).float()]
            M_power=M
            for _ in range(args.n_rw-1):
                M_power=M_power*M
                SE_rw.append(torch.from_numpy(M_power.diagonal()).float())
            SE_rw=torch.stack(SE_rw,dim=-1)

            g['stc_enc'] = SE_rw

    elif type_init == 'dg':
        for g in gs:
            # PE_degree
            g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, args.n_dg)
            SE_dg = torch.zeros([g.num_nodes, args.n_dg])
            for i in range(len(g_dg)):
                SE_dg[i,int(g_dg[i]-1)] = 1

            g['stc_enc'] = SE_dg

    elif type_init == 'rw_dg':
        for g in gs:
            # SE_rw

            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            
            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

            Dinv = sp.diags(D)

            RW = A*Dinv
            
            M = RW
            
            SE = [torch.from_numpy(M.diagonal()).float()]
            
            M_power = M
            
            for _ in range(args.n_rw-1):
                M_power = M_power*M
                SE.append(torch.from_numpy(M_power.diagonal()).float())
                
            SE_rw=torch.stack(SE,dim=-1)

          
            g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, args.n_dg)
            
           
            SE_dg = torch.zeros([g.num_nodes, args.n_dg])
            
            for i in range(len(g_dg)):

                SE_dg[i,int(g_dg[i]-1)] = 1

            g['stc_enc'] = torch.cat([SE_rw, SE_dg], dim=1)

    return gs


def init_structure_encoding_node(args, g):

    # SE_rw
    A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
    D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

    Dinv=sp.diags(D)
    RW=A*Dinv
    M=RW

    SE=[torch.from_numpy(M.diagonal()).float()]
    M_power=M
    for _ in range(args.n_rw-1):
        M_power=M_power*M
        SE.append(torch.from_numpy(M_power.diagonal()).float())
    SE_rw=torch.stack(SE,dim=-1)

    # PE_degree
    g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, args.n_dg)
    SE_dg = torch.zeros([g.num_nodes, args.n_dg])
    for i in range(len(g_dg)):
        SE_dg[i,int(g_dg[i]-1)] = 1

    g['stc_enc'] = torch.cat([SE_rw, SE_dg], dim=1)

    return g


def compute_degree_similarity(d_a, d_b):
    return 1 / (1 + abs(d_a - d_b))

def compute_jaccard_similarity(neighbors_a, neighbors_b):


    intersection = len(neighbors_a.intersection(neighbors_b))
    union = len(neighbors_a.union(neighbors_b))

    j_sim = 0.0 if union == 0 else intersection / float(union)
    return j_sim
        
def init_fedcap(args, g):
    # random walk embedding
    A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
    D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

    Dinv=sp.diags(D)
    RW=A*Dinv
    M=RW

    SE=[torch.from_numpy(M.diagonal()).float()]
    M_power=M
    for _ in range(args.n_rw-1):
        M_power=M_power*M
        SE.append(torch.from_numpy(M_power.diagonal()).float())
        
    random_emb = torch.stack(SE,dim=-1)

    g_dg = degree(g.edge_index[0], num_nodes=g.num_nodes).numpy()
    

    neighbors = {}
    for i in range(g.num_nodes):
        outgoing_neighbors = set(g.edge_index[1][g.edge_index[0] == i].numpy())  
        incoming_neighbors = set(g.edge_index[0][g.edge_index[1] == i].numpy())  
        neighbors[i] = outgoing_neighbors.union(incoming_neighbors)


    similarities_d, similarities_j = {}, {}

    for node in range(g.num_nodes):
        node_degree = g_dg[node]
        node_neighbors = neighbors[node]
        for neighbor in node_neighbors:
            if node < neighbor:
                ## 1. compute degree similarity
                neighbor_degree = g_dg[neighbor]
                similarity = compute_degree_similarity(node_degree, neighbor_degree)
                if node not in similarities_d:
                    similarities_d[node] = {}
                similarities_d[node][neighbor] = similarity
                if neighbor not in similarities_d:
                    similarities_d[neighbor] = {}
                similarities_d[neighbor][node] = similarity

                ## 2. compute jaccard similarity
                similarity2 = compute_jaccard_similarity(neighbors[node], neighbors[neighbor])
                if node not in similarities_j:
                    similarities_j[node] = {}
                similarities_j[node][neighbor] = similarity2
                if neighbor not in similarities_j:
                    similarities_j[neighbor] = {}
                similarities_j[neighbor][node] = similarity2

    all_similarities_d = np.concatenate([list(node_sim.values()) for node_sim in similarities_d.values()])
    all_similarities_j = np.concatenate([list(node_sim.values()) for node_sim in similarities_j.values()])
    
    S_max_d, S_min_d = all_similarities_d.max(), all_similarities_d.min()
    S_max_j, S_min_j = all_similarities_j.max(), all_similarities_j.min()
    

    normalized_similarities_d, normalized_similarities_j = {}, {}
    
    for node in similarities_d:
        normalized_similarities_d[node] = {}
        for neighbor, sim in similarities_d[node].items():
            normalized_similarities_d[node][neighbor] = (sim - S_min_d) / (S_max_d - S_min_d)
    for node in similarities_j:
        normalized_similarities_j[node] = {}
        for neighbor, sim in similarities_j[node].items():
            normalized_similarities_j[node][neighbor] = (sim - S_min_j) / (S_max_j - S_min_j)

    average_normalized_similarity_d, average_normalized_similarity_j = {}, {}
    
    for node in range(g.num_nodes):
        if node in normalized_similarities_d:
            average_normalized_similarity_d[node] = np.mean(list(normalized_similarities_d[node].values()))
        else:
            average_normalized_similarity_d[node] = 0
            
        if node in normalized_similarities_j:
            average_normalized_similarity_j[node] = np.mean(list(normalized_similarities_j[node].values()))
        else:
            average_normalized_similarity_j[node] = 0

    sorted_similarity_values_d = [average_normalized_similarity_d[node] for node in range(g.num_nodes)]
    sorted_similarity_values_j = [average_normalized_similarity_j[node] for node in range(g.num_nodes)]
    
    degree_emb  = torch.tensor(sorted_similarity_values_d, dtype=torch.float32).unsqueeze(1)
    jaccard_emb = torch.tensor(sorted_similarity_values_j, dtype=torch.float32).unsqueeze(1)
    
    g['stc_enc'] = torch.cat([random_emb, degree_emb, jaccard_emb], dim = 1)
    return g
