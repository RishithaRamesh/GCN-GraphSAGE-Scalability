"""Mini-batch GraphSAGE experiments on OGB ogbn-products.

Uses NeighborLoader with bounded fanout for both training and evaluation.
Evaluation uses stochastic sampled neighborhoods (not exact full-neighbor inference).
For a quick smoke test, set --max-train-batches and --eval-max-nodes.
Supports CUDA on Hazel, MPS on Apple Silicon, and CPU fallback.
"""
import argparse
import csv
import os
import random
import time
from pathlib import Path

# OGB processed files from trusted OGB downloads can require this setting.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(in_dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def make_loader(data, indices, fanouts, batch_size, shuffle, workers):
    return NeighborLoader(
        data, input_nodes=indices, num_neighbors=fanouts,
        batch_size=batch_size, shuffle=shuffle, num_workers=workers,
    )


@torch.no_grad()
def evaluate(model, data, split_idx, fanouts, batch_size, device, workers, max_nodes):
    model.eval()
    output = {}
    for split in ('train', 'valid', 'test'):
        indices = split_idx[split]
        if max_nodes > 0:
            indices = indices[:max_nodes]
        loader = make_loader(data, indices, fanouts, batch_size, False, workers)
        correct, loss_sum, count = 0, 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            count_batch = batch.batch_size
            logits = model(batch.x, batch.edge_index)[:count_batch]
            labels = batch.y[:count_batch].view(-1)
            loss_sum += F.cross_entropy(logits, labels, reduction='sum').item()
            correct += (logits.argmax(-1) == labels).sum().item()
            count += count_batch
        output[f'{split}_loss'] = loss_sum / count
        output[f'{split}_acc'] = correct / count
        output[f'{split}_evaluated_nodes'] = count
    return output


def save_outputs(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(str(path) + '.csv')
    png_path = Path(str(path) + '.png')
    with csv_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split in ('train', 'valid', 'test'):
        axes[0].plot([r['epoch'] for r in rows], [100*r[f'{split}_acc'] for r in rows], label=split)
        axes[1].plot([r['epoch'] for r in rows], [r[f'{split}_loss'] for r in rows], label=split)
    axes[0].set(xlabel='Epoch', ylabel='Accuracy (%)', title='Sampled evaluation accuracy')
    axes[1].set(xlabel='Epoch', ylabel='Cross entropy', title='Sampled evaluation loss')
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f'Saved {csv_path} and {png_path}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--neighbors', type=int, nargs='+', default=[10, 10])
    parser.add_argument('--hidden-channels', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-train-batches', type=int, default=0,
                        help='0 = all batches; positive value for a smoke test')
    parser.add_argument('--eval-max-nodes', type=int, default=0,
                        help='0 = entire split; positive value evaluates only the first N nodes')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--dataset-root', default=os.environ.get('OGB_ROOT', 'dataset'))
    parser.add_argument('--output-dir', default='graphsage/results')
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not args.neighbors or any(n < 1 for n in args.neighbors):
        parser.error('epochs, batch size, and neighbors must all be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'auto':
        selected = ('cuda' if torch.cuda.is_available() else
                    'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        selected = args.device
    if selected == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available on this machine')
    if selected == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is not available on this machine')
    device = torch.device(selected)
    print(f'Device: {device}', flush=True)
    print(f'Dataset root: {args.dataset_root}', flush=True)
    dataset = PygNodePropPredDataset(name='ogbn-products', root=args.dataset_root)
    data = dataset[0]  # Keep the large graph in host memory.
    split_idx = dataset.get_idx_split()
    print(f'Graph: {data.num_nodes:,} nodes, {data.num_edges:,} edges', flush=True)
    train_loader = make_loader(data, split_idx['train'], args.neighbors,
                               args.batch_size, True, args.num_workers)
    model = GraphSAGE(data.num_features, args.hidden_channels, dataset.num_classes,
                      len(args.neighbors), args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rows = []
    process = psutil.Process()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        train_sum, train_nodes = 0.0, 0
        for step, batch in enumerate(train_loader):
            if args.max_train_batches and step >= args.max_train_batches:
                break
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)[:batch.batch_size]
            labels = batch.y[:batch.batch_size].view(-1)
            loss = F.cross_entropy(pred, labels)
            loss.backward()
            optimizer.step()
            train_sum += loss.item()*batch.batch_size
            train_nodes += batch.batch_size
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        train_sec = time.perf_counter() - start
        metrics = evaluate(model, data, split_idx, args.neighbors, args.batch_size,
                           device, args.num_workers, args.eval_max_nodes)
        row = {'epoch': epoch, 'optimization_loss': train_sum/train_nodes,
               **metrics, 'train_time_s': train_sec,
               'gpu_peak_allocated_mb': (torch.cuda.max_memory_allocated()/1024**2
                                         if device.type == 'cuda' else float('nan')),
               'gpu_peak_reserved_mb': (torch.cuda.max_memory_reserved()/1024**2
                                        if device.type == 'cuda' else float('nan')),
               'cpu_rss_mb_snapshot': process.memory_info().rss/1024**2,
               'train_target_nodes': train_nodes}
        rows.append(row)
        print(f'Epoch {epoch} | train {metrics["train_acc"]:.2%} | '
              f'valid {metrics["valid_acc"]:.2%} | test {metrics["test_acc"]:.2%} | '
              f'loss {row["optimization_loss"]:.4f} | train {train_sec:.1f}s | '
              f'CUDA peak {row["gpu_peak_allocated_mb"]:.0f} MB', flush=True)
    stem = (f'products_sage_neighbors{"-".join(map(str, args.neighbors))}'
            f'_batch{args.batch_size}_lr{args.lr}_seed{args.seed}'
            f'_epochs{args.epochs}_trainbatches{args.max_train_batches}'
            f'_evalnodes{args.eval_max_nodes}')
    save_outputs(rows, Path(args.output_dir)/stem)
    print('NOTE: Evaluation uses sampled neighborhoods, not exact full-neighbor inference.', flush=True)
    if args.eval_max_nodes:
        print('NOTE: Metrics use only a subset of each split and are smoke-test metrics.', flush=True)
    if args.max_train_batches:
        print('NOTE: Training used only a subset of batches per epoch.', flush=True)


if __name__ == '__main__':
    main()
