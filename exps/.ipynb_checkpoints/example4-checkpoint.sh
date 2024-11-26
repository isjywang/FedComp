export CUDA_VISIBLE_DEVICES=1
python3 main_multiDS.py --mode hete6 --dataset CS --clients 10 --alg gcfl --seed 3  > ./CS/gcfl_10_3hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 10 --alg gcfl --seed 4  > ./CS/gcfl_10_4hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 20 --alg gcfl --seed 3  > ./CS/gcfl_20_3hete6.txt
python3 main_multiDS.py --mode hete6 --dataset CS --clients 20 --alg gcfl --seed 4  > ./CS/gcfl_20_4hete6.txt
