# GCN vs. GraphSAGE: Scalability Analysis

This project explores the differences between **full-batch Graph Convolutional Networks (GCN)** and **mini-batch GraphSAGE** for node classification.

The goal is to understand how model architecture, learning rate, batch size, and neighborhood sampling affect accuracy, convergence, training time, and memory usage.

## 1. Full-Batch GCN

We trained GCN models on the **ogbn-arxiv** citation network to study the effect of model depth and learning rate.

We compared 2, 3, and 4-layer GCNs, along with learning rates of 0.001, 0.005, and 0.01.

The experiments showed that learning rate strongly influences convergence speed, while increasing model depth does not necessarily improve accuracy. The 3-layer GCN achieved approximately **72% test accuracy**.

## 2. Mini-Batch GraphSAGE

We trained GraphSAGE on the larger **ogbn-products** dataset using stochastic neighborhood sampling.

We varied batch sizes (256, 512, 1024) and sampled neighbors per layer ([5,5], [10,10], [15,15]).

The results showed that larger neighborhood samples generally improved validation accuracy but increased training time significantly. Batch size had a smaller effect on training time.

The highest sampled validation accuracy was **90.35%**, using neighbor sampling [15,15] and batch size 512.

Evaluation was performed on sampled neighborhoods and subsets of 10,000 nodes per split, so these results are approximate rather than official full-split OGB benchmark scores.

## 3. Memory and Runtime Comparison

We profiled both models on **ogbn-arxiv**, using the same CPU hardware.

| Metric | GCN | GraphSAGE |
|---|---:|---:|
| Training time/epoch | 1.96 s | 45.61 s |
| Peak process memory | 6708.7 MiB | 2108.8 MiB |

GraphSAGE used approximately **69% less peak process memory** than full-batch GCN.

However, it took longer to train because neighborhood sampling introduces overhead. This illustrates the trade-off between memory efficiency and computation time.

## Conclusion

Full-batch GCN is efficient for smaller graphs that fit into memory. GraphSAGE improves scalability by sampling a limited number of neighbors rather than processing the entire graph at once.

The experiments demonstrate how neighborhood sampling makes large-scale graph learning more memory-efficient, while introducing trade-offs in training time and predictive performance.

Detailed experimental results, convergence plots, and analysis are included in the accompanying report.

## Reproducing the Experiments

The experiments were conducted using Python 3.11, PyTorch, PyTorch Geometric, and OGB. Additional dependencies include NumPy, pandas, matplotlib, and psutil.

Run all commands from the repository's root directory. 

### Task 1: Full-Batch GCN

Example command for the 3-layer GCN:

```bash
python gcn/gnn.py --num_layers 3 --hidden_channels 256 --lr 0.01 --epochs 200 --runs 1
```

To reproduce the depth experiments, change `--num_layers` to `2`, `3`, or `4`, keeping LR at `0.01`.

To reproduce the learning-rate experiments, keep `--num_layers 3` and change `--lr` to `0.001`, `0.005`, or `0.01`.

Results are saved in `gcn/results/`.

### Task 2: Mini-Batch GraphSAGE

Example command for the baseline experiment:

```bash
python graphsage/graphsage_products.py \
  --epochs 10 \
  --batch-size 512 \
  --neighbors 10 10 \
  --eval-max-nodes 10000 \
  --num-workers 0
```

For the batch-size comparison, use `256`, `512`, and `1024` with neighbors `[10,10]`.

For the neighbor-sampling comparison, use `[5,5]`, `[10,10]`, and `[15,15]` with batch size `512`.

All experiments use learning rate `0.003` and random seed `42`. Results are saved in `graphsage/results/`.

Evaluation uses sampled neighborhoods and 10,000 nodes per split, so accuracy values are approximate rather than official OGB leaderboard scores.

### Task 3: Memory and Runtime Profiling

Both models were profiled on the same CPU using `ogbn-arxiv`.

```bash
python profile_task3.py --model gcn --device cpu --epochs 3
python profile_task3.py --model sage --device cpu --epochs 3
```

Profiling results and comparison plots are saved in `task3/results/`.

**Note:** Numerical results may vary slightly depending on hardware, software versions, and random initialization. The submitted CSV files and plots contain the results reported in the accompanying PDF.

### Dataset Setup

The datasets are not included in the submission because of their size. Both are publicly available through the Open Graph Benchmark (OGB).

The experiments use:
- `ogbn-arxiv` for GCN and profiling.
- `ogbn-products` for GraphSAGE.

Both scripts load datasets from the `dataset/` directory by default. If the datasets are not present, the OGB loader will prompt to download them automatically.

For compatibility with newer PyTorch versions, the scripts use `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` when loading trusted OGB dataset files.