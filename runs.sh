#!/bin/bash
sbatch slurm-scripts/dmy-histeq-train.sh
sbatch slurm-scripts/dmy-train.sh
sbatch slurm-scripts/dvalue-histeq-train.sh
sbatch slurm-scripts/dvalue-train.sh
