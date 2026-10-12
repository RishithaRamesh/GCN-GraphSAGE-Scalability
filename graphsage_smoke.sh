#!/bin/bash
#SBATCH --job-name=sage-smoke
#SBATCH --account=csc591038f26_gpu
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=sage_smoke_%j.out

set -euo pipefail
module load conda
source /usr/local/apps/conda/miniconda3/26.3.2/etc/profile.d/conda.sh
conda activate /share/csc591038f26/rsramesh/envs/gnn-hw2
export OGB_ROOT=/share/csc591038f26/rsramesh/datasets
python -u graphsage/graphsage_products.py \
  --epochs 1 --batch-size 256 --neighbors 5 5 \
  --max-train-batches 5 --eval-max-nodes 1000
