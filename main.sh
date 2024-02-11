#!/bin/bash
#SBATCH --job-name=hipp_test
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00
#SBATCH -o slurms/%j.out
# SBATCH --partition=della-gpu
# SBATCH --gres=gpu:1

# for i in {1..10}; do sbatch main.sh; done
source ~/.bashrc
cd /scratch/gpfs/qanguyen/alpha-zero-general
# ./.venv/bin/jupyter nbconvert main.ipynb --to python && RAY_DEDUP_LOGS=0 ./.venv/bin/python main.py --numEps 1 --num_jobs_at_a_time 1 --do_pretrain --startIter 11  --epochs 1 --debug --numIters 12 --max_garside_len 10
# ./.venv/bin/python main.py --numEps 1 --num_jobs_at_a_time 1 --do_pretrain --startIter 11  --epochs 1
# RAY_DEDUP_LOGS=0 ./.venv/bin/python main.py --numEps 1 --num_jobs_at_a_time 1 --do_pretrain --startIter 2  --epochs 10 --numIters 1000 --max_garside_len 100 --playout_cap_randomization_prob 0.1 --cpuct 1.1 --lr 2e-5 --debug
# RAY_DEDUP_LOGS=0 ./.venv/bin/python main.py --numEps 1 --num_jobs_at_a_time 1 --no-do_pretrain --load_checkpoint ./temp/pretrained.pth.tar --startIter 31  --epochs 1 --numIters 1 --max_garside_len 10 --playout_cap_randomization_prob 1.0 --cpuct 1.1 --lr 2e-5 --debug
RAY_DEDUP_LOGS=0 ./.venv/bin/python main.py --numEps 100 --num_jobs_at_a_time 20 --no-do_pretrain --startIter 1  --epochs 10 --numIters 1000 --max_garside_len 100 --playout_cap_randomization_prob 0.25 --cpuct 1.1 --lr 2e-5