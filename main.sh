#!/bin/bash
#SBATCH --job-name=hipp_test
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --time=8:00:00
#SBATCH --output braids-%J.log
#SBATCH -o slurms/%j.out
#SBATCH --partition=della-gpu
#SBATCH --gres=gpu:1

# for i in {1..10}; do sbatch main.sh; done
source ~/.bashrc
cd /scratch/gpfs/qanguyen/alpha-zero-general
# ./.venv/bin/jupyter nbconvert main.ipynb --to python && RAY_DEDUP_LOGS=0 ./.venv/bin/python main.py --numEps 1 --num_jobs_at_a_time 1 --do_pretrain --startIter 11  --epochs 1 --debug --numIters 12 --max_garside_len 10
./.venv/bin/python main.py --numEps 1 --num_jobs_at_a_time 1 --do_pretrain --startIter 11  --epochs 1