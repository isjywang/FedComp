import os
import sys
import argparse
import random
import copy
import time

import torch
from tensorboardX import SummaryWriter
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import setupGC
from training import *
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for node classification.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for inner solver;')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=0)
    parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=5)
    parser.add_argument('--n_rw', type=int, default = 16,
                        help='Size of position encoding (random walk).')
    parser.add_argument('--n_dg', type=int, default=16,
                         help='Size of position encoding (max degree).')
    parser.add_argument('--hidden_str', type=int, default = 8,
                        help='Size of position encoding (random walk).')
    
    parser.add_argument('--check',type=int, default= 0 )
    parser.add_argument('--dataset',type=str, default='Cora')
    parser.add_argument('--clients',type=int, default= 10 )
    parser.add_argument('--mode',type=str, default='disjoint')
    parser.add_argument('--mu',type=float, default='0.01')
    
    parser.add_argument('--repair_fre',type=float, default=0.3)
    parser.add_argument('--warm_epoch',type=float, default=0.7)
    parser.add_argument('--k',type=int, default=8)
    parser.add_argument('--k2',type=int, default=2)

    parser.add_argument('--alg', type=str, default='fedcap',
                        help='Name of algorithms.')

    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--num_rounds', type=int, default=100,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=2,
                        help='number of local epochs;')
    
    begin = time.time()
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    print("now the method is: ", args.alg)
    print("now the dataset is: ", args.dataset)
    print("the clients num is :", args.clients)
    # set device
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # set training config
    set_config(args)
    
    # preparing data
    print("Preparing data ...")
    # splitedData, df_stats, homogeneous_edge = setupGC.prepareData_nodeTask(args)  
    splitedData = setupGC.prepareData_nodeTask(args)  
    print("Done")

    ## for fedstar
    args.n_se = args.n_rw + args.n_dg
    print("weight decay:",args.weight_decay)
    print("local epoch:",args.local_epoch)
    
    init_clients, init_server = setupGC.setup_devices_nodeTask(splitedData, args)
    print("\nDone setting up devices.")

    t_linshi=time.time()
    if args.alg == 'fedcap':
        run_fedcap(args, init_clients, init_server, args.num_rounds, args.local_epoch)
        # run_graphrepair(args, init_clients, init_server, args.num_rounds, args.local_epoch, homogeneous_edge, samp=None)
    elif args.alg == 'fedprox':
        run_fedprox(args, init_clients, init_server, args.num_rounds, args.local_epoch, mu=args.mu)
    elif args.alg == 'gcfl':
        run_gcflplus(args, init_clients, init_server, args.num_rounds, args.local_epoch)
        # run_gcflplus(args, init_clients, init_server, args.num_rounds, args.local_epoch, homogeneous_edge, samp=None)
    elif args.alg == 'local':
        run_local(args, init_clients, init_server, args.num_rounds, args.local_epoch)
    elif args.alg == 'fedstar':
        run_fedstar(args, init_clients, init_server, args.num_rounds, args.local_epoch)
    else: ## fedavg
        run_fedavg(args, init_clients, init_server, args.num_rounds, args.local_epoch)
        # run_train(args, init_clients, init_server, args.num_rounds, args.local_epoch,  homogeneous_edge, samp=None )
    end = time.time()
    print("all time:", end - begin)
    print("linshi time:",end-t_linshi)
    sys.stdout.flush()
    os._exit(0)


