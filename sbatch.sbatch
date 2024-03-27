#!/bin/bash
#SBATCH --job-name="Dataset Rendering"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:0,VRAM:12G
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun --unbuffered /home/stud/vogm/kubric/kubric_env/bin/python3 examples/render.py --logging_level=WARNING
