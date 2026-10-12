"""Mini-batch GraphSAGE sanity-check experiments on ogbn-arxiv.

Run from repository root:
    python graphsage/graphsage.py --epochs 10 --batch-size 512 --neighbors 10 10

For HW2's large-graph results, use a separate ogbn-products pipeline with
scalable evaluation rather than full-graph inference as implemented here.
"""
import argparse
import csv
import os
import random
import time
from pathlib import Path

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
from torch_geometric.utils import to_undirected


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            inputs = in_channels if i == 0 else hidden_channels
            outputs = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(inputs, outputs))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, data, split_idx):
    """Full-graph evaluation: suitable for arxiv smoke tests, NOT products."""
    model.eval()
    logits = model(data.x, data.edge_index)
    y = data.y.view(-1)
    metrics = {}
    for name in ('train', 'valid', 'test'):
        idx = split_idx[name]
        subset_logits = logits[idx]
        target = y[idx]
        metrics[f'{name}_loss'] = F.cross_entropy(subset_logits, target).item()
        metrics[f'{name}_acc'] = (subset_logits.argmax(-1) == target).float().mean().item()
    return metrics


def write_outputs(rows, base):
    base.parent.mkdir(parents=True, exist_ok=True)
    # Concatenate the suffix instead of Path.with_suffix; learning rates contain dots.
    csv_path = Path(str(base) + '.csv')
    png_path = Path(str(base) + '.png')
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = [r['epoch'] for r in rows]
    for split in ('train', 'valid', 'test'):
        axes[0].plot(epochs, [r[f'{split}_acc'] * 100 for r in rows], label=split)
        axes[1].plot(epochs, [r[f'{split}_loss'] for r in rows], label=split)
    axes[0].set(ylabel='Accuracy (%)', xlabel='Epoch', title='GraphSAGE Accuracy')
    axes[1].set(ylabel='Cross-entropy loss', xlabel='Epoch', title='GraphSAGE Loss')
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f'Saved: {csv_path} and {png_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--neighbors', nargs='+', type=int, default=[10, 10])
    parser.add_argument('--hidden-channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset-root', type=str, default='gcn/dataset')
    parser.add_argument('--output-dir', type=str, default='graphsage/results')
    args = parser.parse_args()
    if not args.neighbors or any(n <= 0 for n in args.neighbors):
        parser.error('--neighbors requires positive integer counts')
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('--epochs and --batch-size must be positive')

    set_seed(args.seed)
    # Legacy OGB dataset may use torch.load without weights_only; only trust
    # the known official OGB dataset in this directory.
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=args.dataset_root)
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    split_idx = dataset.get_idx_split()

    train_loader = NeighborLoader(
        data, input_nodes=split_idx['train'], num_neighbors=args.neighbors,
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    model = GraphSAGE(data.num_features, args.hidden_channels, dataset.num_classes,
                      len(args.neighbors), args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    process = psutil.Process(os.getpid())
    rows = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        n_targets = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            target_count = batch.batch_size
            loss = F.cross_entropy(logits[:target_count], batch.y[:target_count].view(-1))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * target_count
            n_targets += target_count
        train_time = time.perf_counter() - started
        metrics = evaluate(model, data, split_idx)
        row = {'epoch': epoch, 'optimization_loss': loss_sum / n_targets,
               **metrics, 'train_time_s': train_time,
               'rss_memory_mb': process.memory_info().rss / 1024**2}
        rows.append(row)
        print(f"Epoch {epoch:03d} | train {metrics['train_acc']:.2%} | "
              f"valid {metrics['valid_acc']:.2%} | test {metrics['test_acc']:.2%} | "
              f"loss {row['optimization_loss']:.4f} | train time {train_time:.1f}s")

    fanout = '-'.join(map(str, args.neighbors))
    stem = (f'arxiv_sage_layers{len(args.neighbors)}_neighbors{fanout}'
            f'_batch{args.batch_size}_lr{args.lr}_hidden{args.hidden_channels}'
            f'_seed{args.seed}_epochs{args.epochs}')
    write_outputs(rows, Path(args.output_dir) / stem)
    best = max(rows, key=lambda r: r['valid_acc'])
    print(f"Best validation epoch: {best['epoch']}; "
          f"validation: {best['valid_acc']:.2%}; test: {best['test_acc']:.2%}")
    print('Note: RSS memory is a per-epoch snapshot, not peak memory.')


if __name__ == '__main__':
    main()
