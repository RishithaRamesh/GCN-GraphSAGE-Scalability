#!/usr/bin/env python3
"""HW2 Task 3: controlled full-batch GCN vs mini-batch GraphSAGE profiling.

Examples (run from repository root):
  python profile_task3.py --model gcn --device cpu --epochs 3
  python profile_task3.py --model sage --device cpu --epochs 3

Both models use ogbn-arxiv, three layers, 256 hidden channels, the same Mac,
and the same directed-to-undirected graph transformation. Outputs are saved
under task3/results. Comparisons are architecture/paradigm comparisons, not
identical operations or equal-accuracy models.
"""
import argparse
import csv
import os
import threading
import time
from pathlib import Path

# OGB's processed PyG objects are trusted local benchmark files.
# PyTorch 2.6+ defaults torch.load to weights_only=True, which older OGB
# versions do not support. Never enable this for untrusted checkpoints.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psutil
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import to_undirected

MB = 1024 * 1024


class RSSMonitor:
    def __init__(self, sample_seconds=0.02):
        self.process = psutil.Process()
        self.sample_seconds = sample_seconds
        self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                pass
            self._stop.wait(self.sample_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def memory_stats(device):
    if device.type == 'cuda':
        return (torch.cuda.max_memory_allocated() / MB,
                torch.cuda.max_memory_reserved() / MB)
    if device.type == 'mps':
        # These are *snapshots*, not verified peaks across the entire epoch.
        return (torch.mps.current_allocated_memory() / MB,
                torch.mps.driver_allocated_memory() / MB)
    return (float('nan'), float('nan'))


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden, classes, layers):
        super().__init__()
        sizes = [input_dim] + [hidden] * (layers - 1) + [classes]
        self.convs = torch.nn.ModuleList(
            GCNConv(sizes[i], sizes[i + 1], cached=True)
            for i in range(layers)
        )

    def forward(self, x, edges):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edges))
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, edges)


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden, classes, layers):
        super().__init__()
        sizes = [input_dim] + [hidden] * (layers - 1) + [classes]
        self.convs = torch.nn.ModuleList(
            SAGEConv(sizes[i], sizes[i + 1])
            for i in range(layers)
        )

    def forward(self, x, edges):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edges))
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, edges)


def draw_comparison(out_dir):
    records = {}
    for name in ('gcn', 'sage'):
        path = out_dir / f'{name}_profile.csv'
        if path.exists():
            with path.open(newline='') as f:
                records[name] = list(csv.DictReader(f))
    if len(records) != 2:
        return
    # Ensure runs are actually comparable before graphing.
    for field in ('device', 'dataset', 'layers', 'hidden', 'epochs'):
        if records['gcn'][0][field] != records['sage'][0][field]:
            print(f'Comparison chart skipped: {field} differs between runs')
            return
    metrics = [
        ('epoch_time_s', 'Mean training seconds / epoch', 'training_time_comparison.png'),
        ('peak_process_rss_mb', 'Peak process RSS (MiB)', 'memory_comparison.png'),
    ]
    labels = ['Full-batch GCN', 'Mini-batch GraphSAGE']
    for key, title, filename in metrics:
        values = [float(records[m][0][key]) for m in ('gcn', 'sage')]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, values)
        ax.set_ylabel(title)
        ax.set_title(f'ogbn-arxiv | {records["gcn"][0]["device"]} | 3-layer comparison')
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{value:.1f}', ha='center', va='bottom')
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)
        print('Saved', out_dir / filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['gcn', 'sage'], required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--neighbors', type=int, nargs='+', default=[10, 10, 10])
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--dataset-root', default=os.environ.get('OGB_ROOT', 'dataset'))
    parser.add_argument('--output-dir', default='task3/results')
    args = parser.parse_args()
    if args.epochs < 1 or args.layers < 2 or args.hidden < 1 or args.batch_size < 1:
        parser.error('epochs, hidden and batch-size must be positive; layers >= 2')
    if len(args.neighbors) != args.layers:
        parser.error('provide exactly one neighbor count per layer')
    device = torch.device(args.device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is not available')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available')
    torch.manual_seed(42)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Model={args.model} Device={device} Dataset=ogbn-arxiv', flush=True)
    with RSSMonitor() as rss:
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=args.dataset_root)
        graph = dataset[0]
        graph.edge_index = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)
        indices = dataset.get_idx_split()['train']
        model_type = GCN if args.model == 'gcn' else GraphSAGE
        model = model_type(graph.num_features, args.hidden, dataset.num_classes,
                           args.layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        if args.model == 'gcn':
            graph = graph.to(device)
            indices = indices.to(device)
        else:
            loader = NeighborLoader(graph, input_nodes=indices,
                                    num_neighbors=args.neighbors,
                                    batch_size=args.batch_size,
                                    shuffle=True, num_workers=0)
        times = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            sync(device)
            start = time.perf_counter()
            if args.model == 'gcn':
                optimizer.zero_grad(set_to_none=True)
                logits = model(graph.x, graph.edge_index)
                loss = F.cross_entropy(logits[indices], graph.y.view(-1)[indices])
                loss.backward()
                optimizer.step()
                batches = 1
            else:
                losses = []
                batches = 0
                for batch in loader:
                    batch = batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(batch.x, batch.edge_index)
                    loss = F.cross_entropy(
                        logits[:batch.batch_size],
                        batch.y.view(-1)[:batch.batch_size])
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    batches += 1
                loss = torch.tensor(sum(losses) / len(losses))
            sync(device)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f'Epoch {epoch}/{args.epochs} | batches={batches} '
                  f'| loss={loss.item():.4f} | training={elapsed:.2f}s', flush=True)
        accelerator_allocated, accelerator_reserved = memory_stats(device)
    row = {
        'model': args.model, 'dataset': 'ogbn-arxiv', 'device': args.device,
        'layers': args.layers, 'hidden': args.hidden, 'epochs': args.epochs,
        'batch_size': args.batch_size if args.model == 'sage' else 'all nodes',
        'neighbors': '-'.join(map(str, args.neighbors)) if args.model == 'sage' else 'full',
        'epoch_time_s': round(sum(times) / len(times), 4),
        'total_training_time_s': round(sum(times), 4),
        'peak_process_rss_mb': round(rss.peak / MB, 3),
        'accelerator_allocated_snapshot_mb': round(accelerator_allocated, 3),
        'accelerator_reserved_snapshot_mb': round(accelerator_reserved, 3),
        'memory_note': 'RSS peak sampled across loading and training; accelerator values are snapshots, except CUDA allocated/reserved peaks',
    }
    path = out_dir / f'{args.model}_profile.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f'Peak process RSS: {row["peak_process_rss_mb"]:.1f} MiB')
    print(f'Mean training time: {row["epoch_time_s"]:.2f} seconds/epoch')
    print('Saved', path)
    draw_comparison(out_dir)


if __name__ == '__main__':
    main()
