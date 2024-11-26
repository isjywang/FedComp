import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def run_local(args, clients, server, COMMUNICATION_ROUNDS, local_epoch):

    print("begin local train")
    print("all communication rounds: ",COMMUNICATION_ROUNDS)
    print("local training epoch: ",local_epoch)
   
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        
        for client in clients:
            client.local_train(local_epoch)

        if c_round == COMMUNICATION_ROUNDS:
            all_acc = 0
            all_size = 0
            for client in clients:
                acc = client.evaluate()
                all_acc+=acc*client.train_size
                all_size+=client.train_size
            print("average acc: ",all_acc/float(all_size))    

    print("train finished!")

def run_fedstar(args, clients, server, COMMUNICATION_ROUNDS, local_epoch):

    print("begin fedstar train")
    print("all communication rounds: ",COMMUNICATION_ROUNDS)
    print("local training epoch: ",local_epoch)

    for client in clients: ##重要
        client.download_from_server(args, server)
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        
        for client in clients:
            client.local_train(local_epoch) 
        
        server.aggregate_weights_se(clients)
        for client in clients:
            client.download_from_server_se(args, server)

        if c_round == COMMUNICATION_ROUNDS:
            all_acc = 0
            all_size = 0
            for client in clients:
                acc = client.evaluate()
                all_acc+=acc*client.train_size
                all_size+=client.train_size
            print("average acc: ",all_acc/float(all_size))    
    print("train finished!")
    
def run_fedavg(args, clients, server, COMMUNICATION_ROUNDS, local_epoch):
    print("begin fedavg train")
    print("all communication rounds: ",COMMUNICATION_ROUNDS)
    print("local training epoch: ",local_epoch)
    for client in clients: ##重要
        client.download_from_server(args, server)

    all_train_time = 0
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        begin = time.time()
        for client in clients:
            client.local_train(local_epoch)
        end = time.time()
        all_train_time = all_train_time + end - begin

            
        server.aggregate_weights(clients)
        for client in clients:
            client.download_from_server(args, server)


        if c_round == COMMUNICATION_ROUNDS:
            all_acc = 0
            all_size = 0
            for client in clients:
                acc = client.evaluate()
                all_acc+=acc*client.train_size
                all_size+=client.train_size
                # acc_all.append(acc)
            print("average acc: ",all_acc/float(all_size))
    print("train finished!")
    print("all train time:",all_train_time)



def run_fedcap(args, clients, server, COMMUNICATION_ROUNDS, local_epoch):
    augmentation_time, aggregation_time = 0, 0
    train_time = 0
    print("begin FedCap train")
    print("all communication rounds: ",COMMUNICATION_ROUNDS)
    print("local training epoch: ",local_epoch)
    for client in clients:
        client.download_from_server(args, server)

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        t1=time.time()
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        for client in clients:
            client.local_train(local_epoch)
        t2=time.time()
        train_time = train_time + t2 - t1
                
        b1=time.time()
        if c_round*local_epoch < int(args.warm_epoch*COMMUNICATION_ROUNDS):

            server.aggregate_weights(clients)
            for client in clients:
                client.download_from_server(args, server)
        else:
            server.fedcap_aggregate(clients)
            for client in clients:
                client.download_from_server_gr(args, server)
        b2=time.time()
        aggregation_time = aggregation_time + b2 - b1
        # server.aggregate_weights(clients)
        # for client in clients:
        #     client.download_from_server(args, server)

        if c_round == COMMUNICATION_ROUNDS:
            all_acc = 0
            all_size = 0

            for client in clients:
                acc = client.evaluate()
                all_acc+=acc*client.train_size
                all_size+=client.train_size
            print("average acc: ",all_acc/float(all_size))
            # a.append(acc)
        # print(a)
        # print(c_round,"after:",all_acc/float(all_size))
        # print(c_round,sum(all_check)/float(len(all_check)))
        # if c_round == COMMUNICATION_ROUNDS:
        #     all_acc = []
        #     for client in clients:
        #         acc = client.evaluate()
        #         all_acc.append(acc)
        #     print("average acc: ",sum(all_acc)/float(len(all_acc)))
        
        # if c_round>1 and c_round<COMMUNICATION_ROUNDS and c_round % int(args.repair_fre*COMMUNICATION_ROUNDS) == 0: ## repair all subgraph 
        #     b3 = time.time()
        #     for client in clients:
        #         client.repair_subgraph(stage=c_round/float(COMMUNICATION_ROUNDS)*len(clients)/10)
        #     b4 = time.time()
        #     augmentation_time = augmentation_time + b4 - b3
            
    print("train finished!")
    print("gnn train time:",sum([client.gnn_time for client in clients ]))
    print("str train time:",sum([client.str_time for client in clients ]))
    print("train time:",train_time)
    print(augmentation_time, aggregation_time)
    # print("acc:", sum(begin_agg)/len(begin_agg), sum(after_agg)/len(after_agg))


def run_fedprox(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, mu=0.01):
    mu = 1e-6 if args.dataset in ['CoraFull','CS','Physics'] else args.mu
    print("begin fedprox train")
    print("all communication rounds: ",COMMUNICATION_ROUNDS)
    print("local training epoch: ",local_epoch)
    for client in clients: ##重要
        client.download_from_server(args, server)
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")        

        # if args.alg == 'fedstar':
        #     server.aggregate_weights_se(clients)
        # else:
        
        for client in clients:
            client.local_train_prox(local_epoch, mu)
            
        server.aggregate_weights(clients)
        for client in clients:
            client.download_from_server(args, server)
            client.cache_weights()
        # if c_round == COMMUNICATION_ROUNDS:    
        all_acc = 0
        all_size = 0

        for client in clients:
            acc = client.evaluate()
            all_acc+=acc*client.train_size
            all_size+=client.train_size
        print("average acc: ",all_acc/float(all_size))   
    print("fedprox train finished!")



def run_gcflplus(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1=0.01, EPS_2=0.1, seq_length=10, standardize=False):
    EPS_1, EPS_2 = 1, 3 ## change for node classification task
    print("begin gcflplus train")
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
    seqs_grads = {c.id:[] for c in clients}
    begin_agg = []
    after_agg = []
    print("all communication rounds: ",COMMUNICATION_ROUNDS)
    print("local training epoch: ",local_epoch)
    for client in clients: ##重要
        client.download_from_server(args, server)

    
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")        
        if c_round == 1:
            for client in clients:
                client.download_from_server(args, server)
        
        for client in clients:
            client.compute_weight_update(local_epoch)
            client.reset()

            seqs_grads[client.id].append(client.convGradsNorm)
            
        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            print(mean_norm,max_norm)
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20 \
                    and all(len(value) >= seq_length for value in seqs_grads.values()):

                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances)-dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
        
        server.aggregate_clusterwise(client_clusters)
        # if c_round == COMMUNICATION_ROUNDS:
        acc_clients = [client.evaluate()*client.train_size for client in clients]
        all_size = [client.train_size for client in clients]
        print("average acc: ",sum(acc_clients)/float(sum(all_size)) )  

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)
    print("GCFL Plus train finished!")




def run_fedprox_observation(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0, mu=0.01):

    print("all communication rounds: ",COMMUNICATION_ROUNDS)
    print("local training epoch: ",local_epoch)
    for client in clients: ##重要
        client.download_from_server(args, server)
    for c_round in range(1, 201):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")        

        # if args.alg == 'fedstar':
        #     server.aggregate_weights_se(clients)
        # else:
        
        for client in clients:
            client.local_train_prox(1, mu)
        all_acc = []
        for client in clients:
            acc = client.evaluate()
            all_acc.append(acc)
        print("after train, average acc: ",sum(all_acc)/float(len(all_acc))) 
        
        if c_round % 2 ==0:
            server.aggregate_weights(clients)
            for client in clients:
                client.download_from_server(args, server)
                client.cache_weights()
                
            all_acc = []
            for client in clients:
                acc = client.evaluate()
                all_acc.append(acc)
            print("after agg, average acc: ",sum(all_acc)/float(len(all_acc)))       

    print("fedprox train finished!")

def run_gcflplus_observation(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0, EPS_1=0.01, EPS_2=0.1, seq_length=10, standardize=False):
    
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
    seqs_grads = {c.id:[] for c in clients}
    
    print("all communication rounds: ",COMMUNICATION_ROUNDS)
    print("local training epoch: ",local_epoch)
    for client in clients: ##重要
        client.download_from_server(args, server)

    
    for c_round in range(1, 101):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")        
        if c_round == 1:
            for client in clients:
                client.download_from_server(args, server)
        # if args.alg == 'fedstar':
        #     server.aggregate_weights_se(clients)
        # else:
        acc_1,acc_2=[],[]
        for client in clients:
            client.compute_weight_update_1(1)
            acc_1.append(client.evaluate())
            client.compute_weight_update_2(1)
            acc_2.append(client.evaluate())
            client.reset()
            seqs_grads[client.id].append(client.convGradsNorm)
        print("after train, average acc: ",sum(acc_1)/float(len(acc_1)))  
        print("after train, average acc: ",sum(acc_2)/float(len(acc_2)))  
        
        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20 \
                    and all(len(value) >= seq_length for value in seqs_grads.values()):

                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances)-dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
        
        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate() for client in clients]

        print("after agg, average acc: ",sum(acc_clients)/float(len(acc_clients)))  

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    print("GCFL Plus train finished!")



def run_train_o2(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, homogeneous_edge, samp=None, frac=1.0):

    print("all communication rounds: ",COMMUNICATION_ROUNDS)
    print("local training epoch: ",local_epoch)
    for client in clients: ##重要
        client.download_from_server(args, server)
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):

        if c_round % 10 == 0:
            all_homogeneous, now_homogeneous = 0, 0
            for client in clients:
                all_h, exist_h = client.check_homogeneous_2(homogeneous_edge)
                all_homogeneous = all_homogeneous + all_h
                now_homogeneous = now_homogeneous + exist_h
            print("original homogeneous edge keep rate: ", now_homogeneous/float(all_homogeneous))
            
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")        

        # if args.alg == 'fedstar':
        #     server.aggregate_weights_se(clients)
        # else:
        
        for client in clients:
            client.local_train(local_epoch)
            
        server.aggregate_weights(clients)
        for client in clients:
            client.download_from_server(args, server)
        all_acc = []
        for client in clients:
            acc = client.evaluate()
            all_acc.append(acc)
        print("average acc: ",sum(all_acc)/float(len(all_acc)))       

    print("train finished!")



def run_graphrepair_o2(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, homogeneous_edge, samp=None, frac=1.0):

    print("all communication rounds: ",COMMUNICATION_ROUNDS)
    print("local training epoch: ",local_epoch)
    for client in clients: ##重要
        client.download_from_server(args, server)


    for c_round in range(1, COMMUNICATION_ROUNDS + 1):

        if c_round % 10 == 0:
            all_homogeneous, now_homogeneous = 0, 0
            for client in clients:
                all_h, exist_h = client.check_homogeneous_2(homogeneous_edge)
                all_homogeneous = all_homogeneous + all_h
                now_homogeneous = now_homogeneous + exist_h
            print("original homogeneous edge keep rate: ", now_homogeneous/float(all_homogeneous))
            
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        for client in clients:
            client.local_train(local_epoch)
        
        if c_round*local_epoch < int(args.warm_epoch*COMMUNICATION_ROUNDS):
            server.aggregate_weights(clients)
            for client in clients:
                client.download_from_server(args, server)
        else:
            server.graphrepair_aggregate(clients)
            for client in clients:
                client.download_from_server_gr(args, server)

        all_acc = []
        for client in clients:
            acc = client.evaluate()
            all_acc.append(acc)
        print("average acc: ",sum(all_acc)/float(len(all_acc)))
        # print("after agg, average acc: ",sum(all_acc)/float(len(all_acc))) 
        
        if c_round>1 and c_round % int(args.repair_fre*COMMUNICATION_ROUNDS) == 0: ## repair all subgraph 
            for client in clients:
                client.repair_subgraph()
    print("train finished!")




def run_gcflplus_o2(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, homogeneous_edge, samp=None, frac=1.0, EPS_1=0.01, EPS_2=0.1, seq_length=10, standardize=False):
    
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
    seqs_grads = {c.id:[] for c in clients}
    
    print("all communication rounds: ",COMMUNICATION_ROUNDS)
    print("local training epoch: ",local_epoch)
    for client in clients: ##重要
        client.download_from_server(args, server)

    
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):


        if c_round % 10 == 0:
            all_homogeneous, now_homogeneous = 0, 0
            for client in clients:
                all_h, exist_h = client.check_homogeneous_2(homogeneous_edge)
                all_homogeneous = all_homogeneous + all_h
                now_homogeneous = now_homogeneous + exist_h
            print("original homogeneous edge keep rate: ", now_homogeneous/float(all_homogeneous))
            
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")        
        if c_round == 1:
            for client in clients:
                client.download_from_server(args, server)
        # if args.alg == 'fedstar':
        #     server.aggregate_weights_se(clients)
        # else:
        
        for client in clients:
            client.compute_weight_update(local_epoch)
            client.reset()

            seqs_grads[client.id].append(client.convGradsNorm)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20 \
                    and all(len(value) >= seq_length for value in seqs_grads.values()):

                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances)-dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
        
        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate() for client in clients]

        print("average acc: ",sum(acc_clients)/float(len(acc_clients)))  

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

             

    print("GCFL Plus train finished!")