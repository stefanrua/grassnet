#!/bin/bash
#SBATCH --job-name=dvalue-histeq-cc-test
#SBATCH --account=project_2005430
#SBATCH --partition=gpusmall
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1,nvme:2

module load pytorch
tar -xf /scratch/project_2005430/ruastefa/grassnet/images.tar -C $LOCAL_SCRATCH
srun python3 -u model.py \
    --batch-size 64 \
    --image-dir $LOCAL_SCRATCH/images/rgb/ \
    --labels labels/dvalue-test.csv \
    --test \
    --histogram-equalization-combined-channels \
    --run-name dvalue-histeq-cc-test \
    --weights out/24-dvalue-histeq-cc-train/w_best.pt \
