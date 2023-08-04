#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --ntasks=40
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=90000mb

#set conda to an env path:
export PATH="/opt/intel/intelpython3/bin:$PATH"
cd /pfs/data5/home/kit/aifb/ho8030/

nvidia-smi

source activate testenv
which python3
wandb login 5412702c3ad751442cbd9ac96d56a4ccbca97f1e

python3 04_py_files/run_kg_nlm.py