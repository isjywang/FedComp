import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph 
from torch_geometric.utils import to_networkx  
import torch_geometric.data as Data 
from torch_geometric.utils import to_undirected 
import networkx as nx  
import time
import numpy as np
import random 

def copy(target, source, keys):
    for name in keys:
        target[name].data = source[name].data.clone()

def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()
        
class Client_GC():
    def __init__(self, model, client_id, train_size, client_graph, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        # self.name = client_name
        self.train_size = train_size
        self.client_graph = client_graph
        self.optimizer = optimizer
        self.args = args
        
        self.dataLoader = DataLoader(dataset=[self.client_graph], batch_size=args.batch_size, shuffle=False)
        
        # 存储模型的参数，以字典的形式，其中键是参数的名称，值是参数的张量。
        self.W = {key: value for key, value in self.model.named_parameters()}
        # 初始化一个与模型参数相同形状的零张量字典。
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        # 缓存
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        self.gconvNames = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

    def download_from_server(self, args, server):
        ## 将服务器上的权重的键（即模型参数的名称）存储在客户端对象的 `gconvNames` 属性中。
        self.gconvNames = server.W.keys()
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()
            
    def download_from_server_se(self, args, server):
        self.gconvNames = server.W.keys()
        for k in server.W:
            if '_s' in k:
                self.W[k].data = server.W[k].data.clone()            
                
    def cache_weights(self): ##这个方法的目的是在客户端缓存当前的模型参数，以备后续需要进行比较或者回滚操作时使用。
        for name in self.W.keys(): ## 对于模型的每个参数名称 `name`
            ## 将当前参数 `self.W[name]` 的数据克隆并赋值给对应的缓存参数 `self.W_old[name]`。
            self.W_old[name].data = self.W[name].data.clone()

            
            
    def reset(self): ##将模型参数重置为之前缓存的数值。
        copy(target=self.W, source=self.W_old, keys=self.gconvNames)

    def local_train(self, local_epoch):
        """ For self-train & FedAvg """
        train_stats = train_gc(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device, self.id)

    def local_train_prox(self, local_epoch, mu):
        """ For FedProx """
        train_stats = train_gc_prox(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device, self.id, self.gconvNames, self.W, mu, self.W_old)

        # self.weightsNorm = torch.norm(flatten(self.W)).item()

        # weights_conv = {key: self.W[key] for key in self.gconvNames}
        # self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        # grads = {key: value.grad for key, value in self.W.items()}
        # self.gradsNorm = torch.norm(flatten(grads)).item()

        # grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        # self.convGradsNorm = torch.norm(flatten(grads_conv)).item()
        
    def evaluate(self): ##这个函数的作用是使用测试数据对模型进行评估，并返回评估结果。
        acc = eval_gc_nodeTask(self.model, self.dataLoader, self.args.device)
        # print("client ",self.id," acc:", acc)
        return acc

    def cache_weights(self):
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()

    def compute_weight_update(self, local_epoch):
        """ For GCFL plus"""
        copy(target=self.W_old, source=self.W, keys=self.gconvNames)
        train_stats = train_gc(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device, self.id)

        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        self.weightsNorm = torch.norm(flatten(self.W)).item()
        
        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()
        
        dWs_conv = {key: self.dW[key] for key in self.gconvNames}
        self.convDWsNorm = torch.norm(flatten(dWs_conv)).item()
        
        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()
        
        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def compute_weight_update_1(self, local_epoch):
        """ For GCFL plus observation"""
        copy(target=self.W_old, source=self.W, keys=self.gconvNames)
        train_stats = train_gc(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device, self.id)


    def compute_weight_update_2(self, local_epoch):
        """ For GCFL plus observation"""
        train_stats = train_gc(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device, self.id)

        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        self.weightsNorm = torch.norm(flatten(self.W)).item()
        
        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()
        
        dWs_conv = {key: self.dW[key] for key in self.gconvNames}
        self.convDWsNorm = torch.norm(flatten(dWs_conv)).item()
        
        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()
        
        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()


    def check_homogeneous(self, homogeneous_edge):
        original_homogeneous_edge = homogeneous_edge[self.id]
        
        for _, batch in enumerate(self.dataLoader):
            self.optimizer.zero_grad()
            batch.to(self.args.device)
            embeddings = self.model(batch,observation=True)    

        KNN_edge_index = knn_graph(embeddings, k=10, loop=False)
        begin,end = KNN_edge_index[0], KNN_edge_index[1]

        num = 0
        for i in range(len(begin)):
            b,e = int(begin[i]), int(end[i])
            if [b,e] in original_homogeneous_edge or [e,b] in original_homogeneous_edge:
                num = num + 1
        return len(original_homogeneous_edge), num
            
    def check_homogeneous_2(self):
        # original_homogeneous_edge = homogeneous_edge[self.id]
        
        for _, batch in enumerate(self.dataLoader):
            self.optimizer.zero_grad()
            batch.to(self.args.device)
            label = batch.y
            embeddings = self.model(batch)    

        KNN_edge_index = knn_graph(embeddings, k=self.args.k2, loop=False)
        begin,end = KNN_edge_index[0], KNN_edge_index[1]

        num = 0
        for i in range(len(begin)):
            b,e = begin[i], end[i]
            if label[b] == label[e]:
                num = num + 1
        return num/float(len(begin))

class Client_gr():
    def __init__(self, model, client_id, train_size, client_graph, optimizer, args, structual_model, structual_optimizer, num_labels):
        self.model = model.to(args.device)
        self.structual_model = structual_model.to(args.device)
        
        self.id = client_id
        self.train_size = train_size
        self.client_graph = client_graph
        self.args = args
        self.k = self.args.k
        # self.k = num_labels
        
        self.optimizer = optimizer
        self.structual_optimizer = structual_optimizer

        self.dataLoader = DataLoader(dataset=[self.client_graph], batch_size=args.batch_size, shuffle=False)
        
        # 存储模型的参数，以字典的形式，其中键是参数的名称，值是参数的张量。
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.W2 = {key: value for key, value in self.structual_model.named_parameters()}

        self.aug_begin = torch.tensor([])
        self.aug_end = torch.tensor([])
        self.node_begin = self.client_graph.edge_index[0].clone()
        self.node_end = self.client_graph.edge_index[1].clone()
        self.gnn_time = 0 
        self.str_time = 0
        
    def connected_component_size(self):  
        return int(len(self.client_graph.y)/10) 

    def check_homogeneous_2(self):
        # original_homogeneous_edge = homogeneous_edge[self.id]
        
        for _, batch in enumerate(self.dataLoader):
            self.optimizer.zero_grad()
            batch.to(self.args.device)
            label = batch.y
            embeddings = self.model(batch)    

        KNN_edge_index = knn_graph(embeddings, k=self.args.k2, loop=False)
        begin,end = KNN_edge_index[0], KNN_edge_index[1]

        num = 0
        for i in range(len(begin)):
            b,e = begin[i], end[i]
            if label[b] == label[e]:
                num = num + 1
        return num/float(len(begin))
        
    def repair_subgraph(self, stage):
        self.structual_model.eval()
        self.model.eval()
        ## get structual embedding, structual/feature based label
        with torch.no_grad():
            for _, batch in enumerate(self.dataLoader):
                batch.to(self.args.device)
                ## 1. get main model's prediction
                pred_fea = self.model(batch, gr=True)
                emb_fea = pred_fea.clone().detach()
                pred_fea = torch.argmax(pred_fea, axis=1)
                
                
                ## 2. get structural model's prediction
                pred_str = self.structual_model(batch, emb_fea)
                embeddings  = pred_str.clone().detach()
                pred_str = torch.argmax(pred_str, axis=1)

        dif_node = torch.nonzero(pred_str != pred_fea).squeeze().cpu().numpy()
 
        ## build KNN Graph in the embedding space of structual model
        KNN_edge_index = knn_graph(embeddings, k=self.k, loop=False)
        KNN_graph = Data.Data(x=self.client_graph.x, y=self.client_graph.y, edge_index=KNN_edge_index)
        
        ## find largest connected components
        G = to_networkx(KNN_graph) 
        G2 = G.to_undirected()
        largest_cc = list(max(nx.connected_components(G2), key=lambda x: self.connected_component_size()) )
     

        edge1 = self.client_graph.edge_index.t()
        edge2 = edge1.flip(1)
        edge3 = torch.cat((edge1, edge2), dim=0)
        edge3_set = torch.unique(edge3, dim=0)
        new_edge = KNN_edge_index.t()
        new_edge_set = torch.unique(new_edge, dim=0).cpu()
        
        edge3_set = edge3_set.tolist()
        edge3_set = set(map(tuple, edge3_set)) 
        mask = torch.tensor([tuple(row.tolist()) not in edge3_set for row in new_edge_set])
        new_edge_check = new_edge_set[mask]


        min_edges = torch.min(new_edge_check, dim=1)[0]  # 每行的最小值
        max_edges = torch.max(new_edge_check, dim=1)[0]  # 每行的最大值
        sorted_new_edge_check = torch.stack((min_edges, max_edges), dim=1)    
   
        
        largest_in = np.zeros(len(self.client_graph.y), dtype=bool)
        dif_check = np.zeros(len(self.client_graph.y), dtype=bool)
        largest_in[largest_cc] = True
        dif_check[dif_node] = True
        ## 1th, find the edges:
        # add_edges_begin, add_edges_end = [], []
        # for e in sorted_new_edge_check:
        #     if largest_in[e[0]] and largest_in[e[1]] and (dif_check[e[0]] or dif_check[e[1]]):
        #         if int(self.client_graph.y[e[0]]) == int(self.client_graph.y[e[1]]):                    
        #             add_edges_begin.append(e[0])
        #             add_edges_end.append(e[1])
        # print("beign",len(add_edges_begin))
        
        sedges = np.array(sorted_new_edge_check)
        mask_largest_in = largest_in[sedges[:, 0]] & largest_in[sedges[:, 1]]
        mask_dif_check = dif_check[sedges[:, 0]] | dif_check[sedges[:, 1]]
        y_array = np.array(self.client_graph.y)
        same_class_mask = y_array[sedges[:, 0]] == y_array[sedges[:, 1]]

        # 将条件组合起来
        mask_condition = mask_largest_in & mask_dif_check & same_class_mask
        
        # 根据条件筛选出符合要求的边
        add_edges_begin = sedges[mask_condition, 0]
        add_edges_end = sedges[mask_condition, 1]

        # print("after",len(add_edges_begin2))        


        # 2th, random delete edges:
        # print(len(self.aug_begin)/len(self.node_begin))
        if len(self.aug_begin)/len(self.node_begin)>1/2:
            distances = torch.norm(emb_fea[self.aug_begin.to(torch.int64)] - emb_fea[self.aug_end.to(torch.int64)], dim=1).cpu().numpy()
            drop_edges_index = torch.tensor(np.random.choice(len(self.aug_begin), size = int(len(self.aug_begin)*0.3), p=distances/np.sum(distances), replace=False))
            mask = torch.ones(self.aug_begin.size(0), dtype=torch.bool)
            mask[drop_edges_index] = False
            self.aug_begin, self.aug_end = self.aug_begin[mask], self.aug_end[mask] 


        ## 3th, add new edges
        self.aug_begin = torch.cat((self.aug_begin, torch.tensor(add_edges_begin)))
        self.aug_end =   torch.cat((self.aug_end,   torch.tensor(add_edges_end)))

        origin_begin, origin_end = self.node_begin.clone(), self.node_end.clone()

        new_begin = torch.cat((origin_begin, self.aug_begin))
        new_end = torch.cat((origin_end, self.aug_end))

        
        self.client_graph.edge_index = torch.stack((new_begin, new_end)).to(torch.int64)
        
        self.dataLoader = DataLoader(dataset=[self.client_graph], batch_size=self.args.batch_size, shuffle=False)

        
    def download_from_server(self, args, server):
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()

        for q in server.W2:
            self.W2[q].data = server.W2[q].data.clone()
            
    def download_from_server_gr(self, args, server):
        
        for k in server.all_W[self.id]:
            self.W[k].data = server.all_W[self.id][k].data.clone()
        for q in server.W2:
            self.W2[q].data = server.W2[q].data.clone()
            
    def local_train(self, local_epoch):
        # train_stats = train_gc(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device, self.id)
        # train_stats_2 = train_gc(self.structual_model, self.dataLoader, self.structual_optimizer, local_epoch, self.args.device, self.id)
        train_stats, gnn_time, str_time = train_fedcap(self.model, self.structual_model, self.dataLoader, self.optimizer, self.structual_optimizer, local_epoch, self.args.device, self.id)
        self.gnn_time = self.gnn_time + gnn_time
        self.str_time = self.str_time + str_time
        
        
    def evaluate(self): ##这个函数的作用是使用测试数据对模型进行评估，并返回评估结果。
        acc = eval_gc_nodeTask(self.model, self.dataLoader, self.args.device)
        # acc = eval_gc_nodeTask(self.structual_model, self.dataLoader, self.args.device)
        # print("client ",self.id," acc:", acc)
        return acc

    # def check_homogeneous(self, homogeneous_edge):
    #     original_homogeneous_edge = homogeneous_edge[self.id]
        
    #     for _, batch in enumerate(self.dataLoader):
    #         self.optimizer.zero_grad()
    #         batch.to(self.args.device)
    #         embeddings = self.model(batch,observation=True)    

    #     KNN_edge_index = knn_graph(embeddings, k=10, loop=False)
    #     begin,end = KNN_edge_index[0], KNN_edge_index[1]

    #     num = 0
    #     for i in range(len(begin)):
    #         b,e = int(begin[i]), int(end[i])
    #         if [b,e] in original_homogeneous_edge or [e,b] in original_homogeneous_edge:
    #             num = num + 1
    #     return len(original_homogeneous_edge), num


    # def check_homogeneous_2(self, homogeneous_edge):
    #     original_homogeneous_edge = homogeneous_edge[self.id]
        
    #     for _, batch in enumerate(self.dataLoader):
    #         self.optimizer.zero_grad()
    #         batch.to(self.args.device)
    #         label = batch.y
    #         embeddings = self.model(batch,observation=True)    

    #     KNN_edge_index = knn_graph(embeddings, k=self.args.k2, loop=False)
    #     begin,end = KNN_edge_index[0], KNN_edge_index[1]

    #     num = 0
    #     for i in range(len(begin)):
    #         b,e = begin[i], end[i]
    #         if label[b] == label[e]:
    #             num = num + 1
    #     return len(begin), num




def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])
    
def _prox_term(model, gconvNames, Wt):
    prox = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        # only add the prox term for sharing layers (gConv)
        if name in gconvNames:
            prox = prox + torch.norm(param - Wt[name]).pow(2)
    return prox
    
def calc_gradsNorm(gconvNames, Ws):
    grads_conv = {k: Ws[k].grad for k in gconvNames}
    convGradsNorm = torch.norm(flatten(grads_conv)).item()
    return convGradsNorm

def train_gc(model, dataloaders, optimizer, local_epoch, device, client_id, is_less=False):
    acc_test = []
    for epoch in range(local_epoch):
        model.train()
        for _, batch in enumerate(dataloaders):
            optimizer.zero_grad()
            batch.to(device)
            pred = model(batch)
            if is_less:
                new_mask = []
                for i in range(len(batch.train_mask)):
                    if batch.train_mask[i]:
                        new_mask.append(True)
                    else:
                        new_mask.append(False)
                a1 = [i for i in range(len(batch.train_mask)) if batch.train_mask[i]]
                ratio = 1 ## 1-ratio是训练比例
                b = random.sample(a1,int(ratio*len(a1)))
                for i in b:
                    new_mask[i]=False
                new_mask = torch.tensor(new_mask)
                new_mask = new_mask.cuda()
                # a2 = [i for i in range(len(new_mask)) if new_mask[i]]      
                # print(len(a1),len(a2))
                loss = model.loss(pred[new_mask], batch.y[new_mask])
                loss.backward()
                optimizer.step()
            else:
                loss = model.loss(pred[batch.train_mask], batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()                
            
        acc_t = eval_gc_nodeTask(model, dataloaders, device)
        acc_test.append(acc_t)

    return {'trainingLosses': [], 'trainingAccs': [], 'valLosses': [], 'valAccs': [],
            'testLosses': [], 'testAccs': acc_test}


def train_fedcap(model, str_model, dataloaders, opt, opt_str, local_epoch, device, client_id):
    acc_test = []
    gnn_time, str_time = 0, 0
    for epoch in range(local_epoch):
        model.train()
        str_model.train()
        for _, batch in enumerate(dataloaders):
            ## 1. train main model and get emb
            begin=time.time()
            opt.zero_grad()
            batch.to(device)
            pred = model(batch)
            loss = model.loss(pred[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            opt.step()     
            end = time.time()
            gnn_time = gnn_time + end - begin
            
            emb_main = model(batch)
            ## 2. get str emb
            opt_str.zero_grad()
            emb_a = emb_main.clone().detach()

            begin2=time.time()
            pred2 = str_model(batch, emb_a)
            loss2 = str_model.loss(pred2[batch.train_mask], batch.y[batch.train_mask])
            loss2.backward()
            opt_str.step()          
            end2 = time.time()
            str_time = str_time + end2 - end
            
            ## 3. get input for concat-model and train
            # opt_concat.zero_grad()
            
            # embstr = emb_str.clone().detach()
            # inputf = torch.cat((embmain, embstr),-1)
            # inputf.requires_grad_()
            
            # predf = concat_model(inputf)
            # loss2 = concat_model.loss(predf[batch.train_mask], batch.y[batch.train_mask])
            # loss2.backward()
            # opt_concat.step()

            ## 4. train str model
            # emb_str.backward( inputf.grad[ : , -emb_str.shape[1] : ] )
            # opt_str.step()
            
        # acc_t = eval_gc_nodeTask(model, dataloaders, device)
        # acc_test.append(acc_t)
    return {'trainingLosses': [], 'trainingAccs': [], 'valLosses': [], 'valAccs': [],
            'testLosses': [], 'testAccs': []}, gnn_time, str_time
            # if is_less:
            #     new_mask = []
            #     for i in range(len(batch.train_mask)):
            #         if batch.train_mask[i]:
            #             new_mask.append(True)
            #         else:
            #             new_mask.append(False)
            #     a1 = [i for i in range(len(batch.train_mask)) if batch.train_mask[i]]
            #     ratio = 1 ## 1-ratio是训练比例
            #     b = random.sample(a1,int(ratio*len(a1)))
            #     for i in b:
            #         new_mask[i]=False
            #     new_mask = torch.tensor(new_mask)
            #     new_mask = new_mask.cuda()
            #     # a2 = [i for i in range(len(new_mask)) if new_mask[i]]      
            #     # print(len(a1),len(a2))
            #     loss = model.loss(pred[new_mask], batch.y[new_mask])
            #     loss.backward()
            #     optimizer.step()
            # else:

    
def train_gc_prox(model, dataloaders, optimizer, local_epoch, device, client_id, gconvNames, Ws, mu, Wt):
    acc_test = []
    convGradsNorm = []
    for epoch in range(local_epoch):
        model.train()
        for _, batch in enumerate(dataloaders):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = model.loss(pred[batch.train_mask], batch.y[batch.train_mask]) + mu / 2. * _prox_term(model, gconvNames, Wt)
            loss.backward()
            optimizer.step()
            
        acc_t = eval_gc_nodeTask(model, dataloaders, device)
        acc_test.append(acc_t)

        convGradsNorm.append(calc_gradsNorm(gconvNames, Ws))

    return {'trainingLosses': [], 'trainingAccs': [], 'valLosses': [], 'valAccs': [],
            'testLosses': [], 'testAccs': acc_test}


def eval_gc(model, test_loader, device):
    model.eval()

    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs

    return total_loss/ngraphs, acc_sum/ngraphs

def eval_gc_nodeTask(model, test_loader, device):
    model.eval()

    # total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)[batch.test_mask]
            label = batch.y[batch.test_mask]
            loss = model.loss(pred, label)
        # total_loss += loss.item() * batch.num_graphs
            # print("pred: ",pred.max(dim=1)[1])
            # print("label: ",label)
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += len(label)

    return acc_sum/ngraphs


def eval_gc_prox(model, test_loader, device, gconvNames, mu, Wt):
    model.eval()

    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label) + mu / 2. * _prox_term(model, gconvNames, Wt)
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs

    return total_loss/ngraphs, acc_sum/ngraphs