export CUDA_VISIBLE_DEVICES=1
python3 main_multiDS.py --dataset CoraFull --clients 50 --seed 0  > ./CoraFull/50_0.txt
python3 main_multiDS.py --dataset CoraFull --clients 50 --seed 1  > ./CoraFull/50_1.txt

python3 main_multiDS.py --dataset CoraFull --clients 50 --alg local --seed 0  > ./CoraFull/local_50_0.txt
python3 main_multiDS.py --dataset CoraFull --clients 50 --alg local --seed 1  > ./CoraFull/local_50_1.txt
python3 main_multiDS.py --dataset CoraFull --clients 50 --alg fedavg --seed 0  > ./CoraFull/fedavg_50_0.txt
python3 main_multiDS.py --dataset CoraFull --clients 50 --alg fedavg --seed 1  > ./CoraFull/fedavg_50_1.txt
python3 main_multiDS.py --dataset CoraFull --clients 50 --alg fedprox --seed 0  > ./CoraFull/fedprox_50_0.txt
python3 main_multiDS.py --dataset CoraFull --clients 50 --alg fedprox --seed 1  > ./CoraFull/fedprox_50_1.txt
python3 main_multiDS.py --dataset CoraFull --clients 50 --alg gcfl --seed 0  > ./CoraFull/gcfl_50_0.txt
python3 main_multiDS.py --dataset CoraFull --clients 50 --alg gcfl --seed 1  > ./CoraFull/gcfl_50_1.txt
python3 main_multiDS.py --dataset CoraFull --clients 50 --alg fedstar --seed 0  > ./CoraFull/fedstar_50_0.txt
python3 main_multiDS.py --dataset CoraFull --clients 50 --alg fedstar --seed 1  > ./CoraFull/fedstar_50_1.txt