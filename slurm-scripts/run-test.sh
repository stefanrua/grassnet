#!/bin/bash
#SBATCH --job-name=dmy
#SBATCH --account=project_2005430
#SBATCH --partition=gputest
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1,nvme:2

module load pytorch
tar -xf /scratch/project_2005430/ruastefa/grassnet/images.tar -C $LOCAL_SCRATCH
srun python3 -u model.py \
    --batch-size 64 \
    --epochs 3 \
    --image-dir $LOCAL_SCRATCH/images/rgb/ \
    --labels labels/tiny.csv \
    --target dmy \
    --save-epoch
