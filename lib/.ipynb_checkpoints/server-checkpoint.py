import torch
import numpy as np
import random
import networkx as nx
from dtaidistance import dtw
from collections import defaultdict, OrderedDict
from utils import *

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])
    
def from_networkx(G, group_node_attrs=None, group_edge_attrs=None): ##将一个NetworkX图转换为PyTorch Geometric数据对象
    import networkx as nx
    from torch_geometric.data import Data

    G = G.to_directed() if not nx.is_directed(G) else G

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue  # Do not load default attributes.
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], torch.Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError, RuntimeError):
                pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data



class Server():
    def __init__(self, model, device):
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.model_cache = []

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()

    def aggregate_weights_per(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            if 'graph_convs' in k:
                self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()

    def aggregate_weights_se(self, selected_clients):
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        # for k in self.W.keys():
        #     self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()
        for k in self.W.keys():
            if '_s' in k:
                self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()

    def aggregate_weights_fe(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            if '_s' not in k:
                self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()


    def compute_pairwise_similarities(self, clients):##计算给定客户端列表中客户端之间的权重相似性
        client_dWs = []
        for client in clients:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            client_dWs.append(dW)
        return pairwise_angles(client_dWs)

    def compute_pairwise_distances(self, seqs, standardize=False):##计算序列之间的动态时间规整（DTW）距离。
        """ computes DTW distances """
        if standardize:
            # standardize to only focus on the trends
            seqs = np.array(seqs)
            seqs = seqs / seqs.std(axis=1).reshape(-1, 1)
            distances = dtw.distance_matrix(seqs)
        else:
            distances = dtw.distance_matrix(seqs)
        return distances

    def min_cut(self, similarity, idc):##使用 NetworkX执行最小割算法, 将一个图分割成两个子图，使得割边的权重之和最小。
        g = nx.Graph()
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                g.add_edge(i, j, weight=similarity[i][j])
        cut, partition = nx.stoer_wagner(g)
        c1 = np.array([idc[x] for x in partition[0]])
        c2 = np.array([idc[x] for x in partition[1]])
        return c1, c2

    def aggregate_clusterwise(self, client_clusters):##对客户端聚类进行聚合操作, 将它们的权重和梯度进行加权平均的聚合操作
        for cluster in client_clusters:
            targs = []
            sours = []
            total_size = 0
            for client in cluster:
                W = {}
                dW = {}
                for k in self.W.keys():
                    W[k] = client.W[k]
                    dW[k] = client.dW[k]
                targs.append(W)
                sours.append((dW, client.train_size))
                total_size += client.train_size
            # pass train_size, and weighted aggregate
            reduce_add_average(targets=targs, sources=sours, total_size=total_size)

    def compute_max_update_norm(self, cluster):
        max_dW = -np.inf
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            update_norm = torch.norm(flatten(dW)).item()
            if update_norm > max_dW:
                max_dW = update_norm
        return max_dW
        # return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    def compute_mean_update_norm(self, cluster):
        cluster_dWs = []
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            cluster_dWs.append(flatten(dW))

        return torch.norm(torch.mean(torch.stack(cluster_dWs), dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs,
                              {name: params[name].data.clone() for name in params},
                              [accuracies[i] for i in idcs])]



class Server_gr():
    def __init__(self, model, device, structual_model, num_features, num_labels, args, all_model):
        self.model = model.to(device)
        self.structual_model = structual_model.to(device)
        self.args = args
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.W2 = {key: value for key, value in self.structual_model.named_parameters()}
        
        self.all_model = all_model ##personalized aggregated model
        self.all_W = []
        for i in range(self.args.clients):
            W = {key: value for key, value in self.all_model[i].named_parameters()}
            self.all_W.append(W)
            
        self.num_features = num_features
        self.num_labels = num_labels
        self.model_cache = []
        self.device = device
        self.random_graph = self.build_random_graph()
        self.random_graph = init_fedcap(self.args, self.random_graph)


    # def cos_similarity(self, x_a, x_b):
    #     return torch.sum(torch.nn.functional.cosine_similarity(x_a, x_b, dim=1))
        
    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()
        for q in self.W2.keys():
            self.W2[q].data = torch.div(torch.sum(torch.stack([torch.mul(client.W2[q].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()
            
    def build_random_graph(self):
        num_nodes, num_graphs = 100, 1
        data = from_networkx(nx.random_partition_graph([num_nodes] * num_graphs, p_in=0.06, p_out=0, seed=666))
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, self.num_features))
        return data

    def get_embedding(self, selected_clients, data):
        all_emb = []
        data = data.to(self.device)
        with torch.no_grad():
            label = self.model(data)
            label = torch.argmax(label, axis=1) 
        data.y = label

        
        for client in selected_clients:
            client.structual_model.eval()
            client.model.eval()
            with torch.no_grad():
                emb_main = client.model(data)
                emb_main = emb_main.clone().detach()
                emb  = client.structual_model(data, emb_main)
                emb      = emb.clone().detach()
                all_emb.append(emb)
        return all_emb

    # def cka_similarity(self, x_a, x_b):  
    # # 计算内积矩阵  
    #     K_a = torch.matmul(x_a, x_a.t())  
    #     K_b = torch.matmul(x_b, x_b.t())  
          
    #     # 计算中心化矩阵  
    #     N = x_a.size(0)
    #     H = torch.eye(N) - torch.ones(N, N) / N
    #     H = H.to(self.device)
    #     centered_K_a = torch.matmul(torch.matmul(H, K_a), H)  
    #     centered_K_b = torch.matmul(torch.matmul(H, K_b), H)
          
    #     # 计算相似度  
    #     trace_cka = torch.trace(centered_K_a @ centered_K_b)  
    #     trace_kka = torch.sqrt(torch.trace(centered_K_a @ centered_K_a) * torch.trace(centered_K_b @ centered_K_b))  
    #     cka = trace_cka / trace_kka  
          
    #     return cka
    def cka_similarity(self, x_a, x_b):  
        # 计算内积矩阵
        K_a = torch.matmul(x_a, x_a.t())  
        K_b = torch.matmul(x_b, x_b.t())  
          
        N = x_a.size(0)
        H = self.H_matrix(N)
    
        # 中心化 K 矩阵
        centered_K_a = H @ K_a @ H
        centered_K_b = H @ K_b @ H
    
        # 计算相似度  
        trace_cka = torch.trace(centered_K_a @ centered_K_b)
        trace_kka = torch.sqrt(
            torch.trace(centered_K_a @ centered_K_a) * torch.trace(centered_K_b @ centered_K_b)
        )
        
        cka = trace_cka / trace_kka
        return cka
    
    def H_matrix(self, N):
    # 创建中心化矩阵 H，避免每次调用都重新创建
        if not hasattr(self, 'H_cache') or self.H_cache.size(0) != N:
            H = torch.eye(N, device=self.device) - torch.ones(N, N, device=self.device) / N
            self.H_cache = H
        return self.H_cache        
        
    # def fedcap_aggregate(self, selected_clients):
    #     random_graph = self.build_random_graph()
    #     embedding = self.get_embedding(selected_clients, random_graph)
        
    #     num_clients = len(selected_clients)
    #     cka = torch.ones((num_clients, num_clients), device=self.device)
        
    #     # 批量计算 CKA 相似度：一次性计算所有客户端之间的 CKA 相似度
    #     embeddings = torch.stack([embedding[i] for i in range(num_clients)])  # 只初始化一次
        
    #     cka_matrix = self.cka_similarity(embeddings, embeddings)  # 使用相同的张量进行批量计算
    #     cka = cka_matrix.cpu()
    
    #     #Currently only supports selecting all clients
    #     for i in range(self.args.clients):
    #         for k in self.all_W[i].keys():
    #             self.all_W[i][k].data = torch.div( torch.sum(torch.stack([torch.mul(selected_clients[client].W[k].data, cka[i][client]) for client in range(len(selected_clients))]), dim=0)  , sum(cka[i]) ).clone()
            
    #     total_size = 0
    #     for client in selected_clients:
    #         total_size += client.train_size
    #     for q in self.W2.keys():
    #         self.W2[q].data = torch.div(torch.sum(torch.stack([torch.mul(client.W2[q].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()
        
    def fedcap_aggregate(self, selected_clients):
        embedding = self.get_embedding(selected_clients, self.random_graph)
        num_clients = len(selected_clients)
        cka = torch.ones((num_clients, num_clients), device=self.device)
        
        for i in range(num_clients):
            for j in range(i+1,num_clients):
                this_cka = float(self.cka_similarity(embedding[i], embedding[j]))
                cka[i][j], cka[j][i] = this_cka, this_cka

        #Currently only supports selecting all clients
        for i in range(self.args.clients):
            for k in self.all_W[i].keys():
                self.all_W[i][k].data = torch.div( torch.sum(torch.stack([torch.mul(selected_clients[client].W[k].data, cka[i][client]) for client in range(len(selected_clients))]), dim=0)  , sum(cka[i]) ).clone()
            
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for q in self.W2.keys():
            self.W2[q].data = torch.div(torch.sum(torch.stack([torch.mul(client.W2[q].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()
        


def reduce_add_average(targets, sources, total_size):
    for target in targets:
        for name in target:
            tmp = torch.div(torch.sum(torch.stack([torch.mul(source[0][name].data, source[1]) for source in sources]), dim=0), total_size).clone()
            target[name].data += tmp