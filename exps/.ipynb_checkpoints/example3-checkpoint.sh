export CUDA_VISIBLE_DEVICES=0
python3 main_multiDS.py --mode hete6 --dataset CS --clients 20 --alg fedavg --seed 3  > ./CS/fedavg_20_3hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 20 --alg fedavg --seed 4  > ./CS/fedavg_20_4hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 10 --alg fedprox --seed 3  > ./CS/fedprox_10_3hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 10 --alg fedprox --seed 4  > ./CS/fedprox_10_4hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 20 --alg fedprox --seed 3  > ./CS/fedprox_20_3hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 20 --alg fedprox --seed 4  > ./CS/fedprox_20_4hete6.txt
