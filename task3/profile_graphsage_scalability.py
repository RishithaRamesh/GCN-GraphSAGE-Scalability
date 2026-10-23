"""CPU GraphSAGE scalability profiling for HW2.

Compares equal-size sampled mini-batches on ogbn-arxiv vs ogbn-products,
and optionally sweeps 3-layer fanouts on ogbn-products.

Run from the repo root, with existing OGB datasets in dataset/.
Only load OGB processed files you trust. Does not deliberately trigger OOM.
This is short-run *step profiling*, NOT full-epoch benchmarking or accuracy reproduction.
"""
import os
# Trusted official OGB cached files may contain PyG Python objects. Must precede torch import.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
import argparse
import csv
import gc
import random
import time
from pathlib import Path
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out, depth):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(depth):
            self.layers.append(SAGEConv(n_in if i == 0 else n_hidden,
                                        n_out if i == depth - 1 else n_hidden))
    def forward(self, x, edges):
        for i, layer in enumerate(self.layers):
            x = layer(x, edges)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        return x

def profile(name, root, fanouts, batch_size, steps, hidden):
    directory = Path(root) / name.replace('-', '_')
    if not (directory / 'processed').is_dir():
        raise FileNotFoundError(f'Cached {name} not found: {directory} (provide the existing parent root)')
    print(f'Loading {name} for fanout {fanouts}...', flush=True)
    dataset = PygNodePropPredDataset(name=name, root=str(root))
    data = dataset[0]
    train_nodes = dataset.get_idx_split()['train'][:batch_size * steps]
    loader = NeighborLoader(data, input_nodes=train_nodes, num_neighbors=fanouts,
                            batch_size=batch_size, shuffle=False, num_workers=0)
    model = GraphSAGE(data.num_features, hidden, dataset.num_classes, len(fanouts))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    process = psutil.Process()
    rows = []
    for step, batch in enumerate(loader, 1):
        rss_before = process.memory_info().rss / (1024**2)
        t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch.x, batch.edge_index)[:batch.batch_size]
        labels = batch.y[:batch.batch_size].view(-1)
        loss = F.cross_entropy(pred, labels)
        loss.backward()
        optimizer.step()
        duration = time.perf_counter() - t0
        rss_after = process.memory_info().rss / (1024**2)
        row = dict(dataset=name, graph_nodes=data.num_nodes, graph_edges=data.num_edges,
                   layers=len(fanouts), batch_size=batch.batch_size,
                   fanout='-'.join(map(str, fanouts)), step=step,
                   sampled_nodes=batch.num_nodes, sampled_edges=batch.num_edges,
                   step_seconds=round(duration, 4), rss_before_mib=round(rss_before, 1),
                   rss_after_mib=round(rss_after, 1), loss=round(loss.item(), 4))
        rows.append(row)
        print(f"  step {step}: sampled={batch.num_nodes:,} nodes, {batch.num_edges:,} edges; "
              f"time={duration:.2f}s; RSS={rss_after:.1f} MiB", flush=True)
    del data, dataset, loader, model, optimizer
    gc.collect()
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', default='dataset')
    p.add_argument('--output', default='task3/results/sage_scalability_cpu.csv')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--fanout', type=int, nargs='+', default=[10, 10])
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--experiment', choices=['cross-dataset', 'fanout', 'both'], default='cross-dataset')
    a = p.parse_args()
    if a.batch_size < 1 or a.steps < 1 or a.hidden < 1 or not a.fanout or min(a.fanout) < 1:
        p.error('batch-size, steps, hidden and fanout values must be positive')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    all_rows = []
    if a.experiment in ('cross-dataset', 'both'):
        for name in ('ogbn-arxiv', 'ogbn-products'):
            all_rows += profile(name, a.dataset_root, a.fanout, a.batch_size, a.steps, a.hidden)
    if a.experiment in ('fanout', 'both'):
        for fanouts in ([5,5,5], [10,5,5], [15,10,5]):
            all_rows += profile('ogbn-products', a.dataset_root, fanouts,
                                a.batch_size, a.steps, a.hidden)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    print(f'\nSaved {len(all_rows)} steps to {out}', flush=True)
    print('NOTE: timings cover forward/backward/optimizer after sampling; ' 
          'RSS is a snapshot including cached datasets, not GPU memory or peak model memory.', flush=True)
    print('NOTE: these short runs measure resource scaling, not convergence or final accuracy.', flush=True)

if __name__ == '__main__':
    main()
