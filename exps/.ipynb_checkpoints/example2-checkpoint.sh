export CUDA_VISIBLE_DEVICES=0
python3 main_multiDS.py --mode hete6 --dataset CS --clients 10 --alg local --seed 3  > ./CS/local_10_3hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 10 --alg local --seed 4  > ./CS/local_10_4hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 20 --alg local --seed 3  > ./CS/local_20_3hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 20 --alg local --seed 4  > ./CS/local_20_4hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 10 --alg fedavg --seed 3  > ./CS/fedavg_10_3hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 10 --alg fedavg --seed 4  > ./CS/fedavg_10_4hete6.txt
