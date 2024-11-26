export CUDA_VISIBLE_DEVICES=0
python3 main_multiDS.py --mode hete6 --dataset CS --clients 10 --alg fedstar --seed 3  > ./CS/fedstar_10_3hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 10 --alg fedstar --seed 4  > ./CS/fedstar_10_4hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 20 --alg fedstar --seed 3  > ./CS/fedstar_20_3hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 20 --alg fedstar --seed 4  > ./CS/fedstar_20_4hete6.txt