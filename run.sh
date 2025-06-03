#!/bin/bash
#SBATCH -N 1 #1 node
#SBATCH --ntasks-per-node=20
##SBATCH --time=3-00:00:00
#SBATCH --job-name="PS Protection Research"
#SBATCH --error=%J.err_
#SBATCH --output=%J.sout_
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2 #2 GPU Cards(1 for single GPU card)
##SBATCH --mem=16GB
#SBATCH --mail-type=all
#SBATCH --mail-user=ommprakash.sahoo.eee20@iitbhu.ac.in


source /scratch/opsahoo.eee20.itbhu/env-setup/bin/activate
conda activate pytorch-gpu
python /home/opsahoo.eee20.itbhu/Master-Thesis/PowerFormer/main.py