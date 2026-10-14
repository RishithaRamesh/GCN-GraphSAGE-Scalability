"""Recreate the four-panel GraphSAGE comparison figure from saved Task 2 CSVs.

Run from the repository root:
    python graphsage/plot_comparison.py

No model training or dataset download is needed.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENTS = {
    'batch 256': 'products_sage_neighbors10-10_batch256_lr0.003_seed42_epochs10_trainbatches0_evalnodes10000.csv',
    'batch 512': 'products_sage_neighbors10-10_batch512_lr0.003_seed42_epochs10_trainbatches0_evalnodes10000.csv',
    'batch 1024': 'products_sage_neighbors10-10_batch1024_lr0.003_seed42_epochs10_trainbatches0_evalnodes10000.csv',
    'fanout [5,5]': 'products_sage_neighbors5-5_batch512_lr0.003_seed42_epochs10_trainbatches0_evalnodes10000.csv',
    'fanout [10,10]': 'products_sage_neighbors10-10_batch512_lr0.003_seed42_epochs10_trainbatches0_evalnodes10000.csv',
    'fanout [15,15]': 'products_sage_neighbors15-15_batch512_lr0.003_seed42_epochs10_trainbatches0_evalnodes10000.csv',
}
REQUIRED_COLUMNS = {'epoch', 'train_acc', 'valid_acc', 'test_acc',
                    'train_loss', 'valid_loss', 'test_loss'}


def read_result(folder: Path, filename: str) -> pd.DataFrame:
    path = folder / filename
    if not path.is_file():
        raise FileNotFoundError(f'Missing experiment results: {path}')
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f'{path.name} missing columns: {sorted(missing)}')
    return df.sort_values('epoch')


def main():
    script_folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=script_folder / 'results',
                        help='Directory containing the five GraphSAGE experiment CSVs')
    parser.add_argument('--output', type=Path,
                        default=script_folder / 'results' / 'graphsage_comparison.png')
    args = parser.parse_args()

    data = {label: read_result(args.results_dir, name)
            for label, name in EXPERIMENTS.items()}
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), constrained_layout=True)

    ax = axes[0, 0]
    for label in ('batch 256', 'batch 512', 'batch 1024'):
        df = data[label]
        ax.plot(df['epoch'], 100 * df['valid_acc'], marker='o', markersize=3, label=label)
    ax.set(title='Batch size (validation)', ylabel='Validation accuracy (%)')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for label in ('fanout [5,5]', 'fanout [10,10]', 'fanout [15,15]'):
        df = data[label]
        ax.plot(df['epoch'], 100 * df['valid_acc'], marker='o', markersize=3, label=label)
    ax.set(title='Neighbor fanout (validation)', ylabel='Validation accuracy (%)')
    ax.legend(fontsize=8)

    baseline = data['batch 512']
    ax = axes[1, 0]
    for split, display in (('train', 'Train'), ('valid', 'Validation'), ('test', 'Test')):
        ax.plot(baseline['epoch'], 100 * baseline[f'{split}_acc'], label=display)
    ax.set(title='Baseline: train / valid / test accuracy',
           ylabel='Accuracy (%)', xlabel='Epoch')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for split, display in (('train', 'Train'), ('valid', 'Validation'), ('test', 'Test')):
        ax.plot(baseline['epoch'], baseline[f'{split}_loss'], label=display)
    ax.set(title='Baseline: train / valid / test loss',
           ylabel='Cross-entropy loss', xlabel='Epoch')
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.23)
        ax.set_xticks(baseline['epoch'][::2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved comparison figure: {args.output}')


if __name__ == '__main__':
    main()
