"""Generate the four-panel Task 1 GCN figure from saved experiment CSVs.

Place this script in gcn/ and run from the repository root:
    python gcn/plot_gcn_comparison.py

Requires pandas and matplotlib. No model training or dataset download is needed.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

EXPERIMENTS = {
    '2-layer, LR 0.01': 'gcn_layers2_lr0.01_hidden256_run1.csv',
    '3-layer, LR 0.01': 'gcn_layers3_lr0.01_hidden256_run1.csv',
    '4-layer, LR 0.01': 'gcn_layers4_lr0.01_hidden256_run1.csv',
    '3-layer, LR 0.001': 'gcn_layers3_lr0.001_hidden256_run1.csv',
    '3-layer, LR 0.005': 'gcn_layers3_lr0.005_hidden256_run1.csv',
}
REQUIRED_COLUMNS = {'epoch', 'train_acc', 'valid_acc', 'test_acc',
                    'train_loss', 'valid_loss', 'test_loss'}


def load_csv(folder: Path, filename: str) -> pd.DataFrame:
    path = folder / filename
    if not path.is_file():
        raise FileNotFoundError(f'Missing CSV: {path}\nCheck that the five confirmed GCN results are in gcn/results/.')
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f'{path} is missing columns: {sorted(missing)}')
    return df.sort_values('epoch')


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=script_dir / 'results')
    parser.add_argument('--output', type=Path,
                        default=script_dir / 'results' / 'gcn_comparison.png')
    args = parser.parse_args()

    data = {name: load_csv(args.results_dir, filename)
            for name, filename in EXPERIMENTS.items()}
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), constrained_layout=True)

    ax = axes[0, 0]
    for name in ('2-layer, LR 0.01', '3-layer, LR 0.01', '4-layer, LR 0.01'):
        df = data[name]
        ax.plot(df['epoch'], df['valid_acc'] * 100, label=name)
    ax.set(title='Effect of depth (validation)', ylabel='Validation accuracy (%)')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for name, label in (('3-layer, LR 0.001', 'LR 0.001'),
                        ('3-layer, LR 0.005', 'LR 0.005'),
                        ('3-layer, LR 0.01', 'LR 0.01')):
        df = data[name]
        ax.plot(df['epoch'], df['valid_acc'] * 100, label=label)
    ax.set(title='Learning-rate sensitivity (validation)', ylabel='Validation accuracy (%)')
    ax.legend(fontsize=8)

    baseline = data['3-layer, LR 0.01']
    ax = axes[1, 0]
    for split, label in (('train', 'Train'), ('valid', 'Validation'), ('test', 'Test')):
        ax.plot(baseline['epoch'], baseline[f'{split}_acc'] * 100, label=label)
    ax.set(title='3-layer, LR 0.01: accuracy', ylabel='Accuracy (%)', xlabel='Epoch')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for split, label in (('train', 'Train'), ('valid', 'Validation'), ('test', 'Test')):
        ax.plot(baseline['epoch'], baseline[f'{split}_loss'], label=label)
    ax.set(title='3-layer, LR 0.01: loss', ylabel='Cross-entropy loss', xlabel='Epoch')
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.23)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved GCN comparison figure: {args.output}')


if __name__ == '__main__':
    main()
