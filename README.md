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