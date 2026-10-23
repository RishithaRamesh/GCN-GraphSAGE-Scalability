# GCN vs. GraphSAGE: Scalability Analysis

This repository contains full-batch GCN experiments, mini-batch GraphSAGE experiments, and memory/runtime scalability profiling.

## Setup

Requirements: Python 3.11, PyTorch, PyTorch Geometric, OGB, NumPy, pandas, matplotlib, and psutil.

```bash
conda activate gnn-hw2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

Run all commands from the **repository root**. The `dataset/` directory is used for OGB datasets (`ogbn-arxiv` and `ogbn-products`). Missing datasets can be downloaded automatically by OGB.

Only enable the environment variable above for trusted dataset files.

## Task 1: Full-Batch GCN

```bash
python gcn/gnn.py \
  --num_layers 3 \
  --hidden_channels 256 \
  --lr 0.01 \
  --epochs 200 \
  --runs 1
```

To reproduce all experiments:

- **Depth:** Set `--num_layers` to `2`, `3`, and `4` with LR `0.01`.
- **Learning rate:** Set `--lr` to `0.001`, `0.005`, and `0.01` with 3 layers.

Results: `gcn/results/`

## Task 2: Mini-Batch GraphSAGE

**Baseline and hyperparameter sweeps:**

```bash
python graphsage/graphsage_products.py \
  --device mps \
  --dataset-root dataset \
  --output-dir graphsage/results \
  --epochs 10 \
  --batch-size 512 \
  --neighbors 10 10 \
  --hidden-channels 128 \
  --lr 0.003 \
  --seed 42 \
  --eval-max-nodes 10000
```

- **Batch sizes:** `256`, `512`, `1024` with neighbors `10 10`.
- **Fanouts:** `5 5`, `10 10`, `15 15` with batch size `512`.
- **30-epoch baseline:** Change `--epochs` to `30`.

**Three-layer benchmark:**

```bash
python graphsage/graphsage_products.py \
  --device cpu \
  --dataset-root dataset \
  --output-dir graphsage/results \
  --epochs 10 \
  --batch-size 1024 \
  --neighbors 15 10 5 \
  --hidden-channels 256 \
  --lr 0.003 \
  --seed 42 \
  --eval-max-nodes 10000
```

Results: `graphsage/results/`

GraphSAGE evaluation uses sampled neighborhoods and 10,000 nodes per split, not exact full-neighborhood OGB inference.

## Task 3: Scalability Profiling

**Cross-dataset scalability:**

```bash
python task3/profile_graphsage_scalability.py \
  --dataset-root dataset \
  --experiment cross-dataset \
  --batch-size 128 \
  --fanout 10 10 \
  --steps 3 \
  --output task3/results/sage_scalability_cpu.csv
```

**Neighbor fanout profiling:**

```bash
python task3/profile_graphsage_scalability.py \
  --dataset-root dataset \
  --experiment fanout \
  --batch-size 128 \
  --steps 3 \
  --output task3/results/sage_fanout_scalability.csv
```

**Full-epoch fanout benchmark:**

```bash
python task3/run_graphsage_fanout_benchmark.py \
  --dataset-root dataset \
  --batch-size 256 \
  --epochs 3
```

**Generate combined scalability figure:**

```bash
python task3/plot_scalability_comparison.py
```

Results: `task3/results/` and `task3/results/fanout_full_epoch/`

## Notes

- Dataset files are excluded from submission due to size.
- Reported results may vary slightly with hardware and library versions.
- CPU memory measurements include dataset and runtime allocations.
- The accompanying PDF report contains the experimental results and analysis.