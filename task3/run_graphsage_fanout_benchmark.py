#!/usr/bin/env python3
"""Run separate CPU GraphSAGE fanout experiments and compile comparable metrics.

Run from the repository root. Requires psutil; invokes existing graphsage_products.py.
Peak RSS is polled OS process RSS, including dataset/runtime allocations, not GPU usage.
"""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

FANOUTS = ((5, 5, 5), (10, 5, 5), (15, 10, 5))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', default='dataset')
    p.add_argument('--script', default='graphsage/graphsage_products.py')
    p.add_argument('--output-dir', default='task3/results/fanout_full_epoch')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--eval-max-nodes', type=int, default=10000,
                   help='0 evaluates complete splits with stochastic neighbor sampling; expensive')
    p.add_argument('--hidden-channels', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fanouts', nargs='*', default=['5,5,5', '10,5,5', '15,10,5'],
                   help='Space-separated entries such as 5,5,5 10,5,5')
    a = p.parse_args()
    script = Path(a.script).resolve()
    root = Path(a.dataset_root).resolve()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not script.exists():
        p.error(f'Script not found: {script}')
    if not (root / 'ogbn_products').exists():
        p.error(f'Cached ogbn-products dataset not found under: {root}')
    results = []
    for fanout_raw in a.fanouts:
        fanout = tuple(map(int, fanout_raw.split(',')))
        if len(fanout) != 3 or any(f <= 0 for f in fanout):
            p.error('Each fanout needs exactly three positive integers')
        label = '-'.join(map(str, fanout))
        stem = (f'products_sage_neighbors{label}_batch{a.batch_size}_lr{a.lr}'
                f'_seed{a.seed}_epochs{a.epochs}_trainbatches0_evalnodes{a.eval_max_nodes}')
        cmd = [sys.executable, str(script), '--device', 'cpu', '--dataset-root', str(root),
               '--output-dir', str(out), '--epochs', str(a.epochs),
               '--batch-size', str(a.batch_size), '--neighbors', *map(str, fanout),
               '--hidden-channels', str(a.hidden_channels), '--lr', str(a.lr),
               '--seed', str(a.seed), '--max-train-batches', '0',
               '--eval-max-nodes', str(a.eval_max_nodes)]
        print('\nRUN', label, '|', ' '.join(cmd), flush=True)
        print('Full-epoch training; larger fanouts can be slow or memory intensive.', flush=True)
        log_path = out / f'fanout_{label}.log'
        peak_rss = 0
        start = time.perf_counter()
        with log_path.open('w') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            ps = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    mem = ps.memory_info().rss
                    peak_rss = max(peak_rss, mem)
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.15)
            rc = proc.wait()
        if rc:
            print(f'FAILED: {label}, exit={rc}. See {log_path}', flush=True)
            break
        path = out / f'{stem}.csv'
        if not path.exists():
            print(f'Missing output CSV: {path}', flush=True)
            break
        with path.open(newline='') as f:
            epochs = list(csv.DictReader(f))
        best = max(epochs, key=lambda r: float(r['valid_acc']))
        mean_seconds = sum(float(r['train_time_s']) for r in epochs) / len(epochs)
        r = {'fanout': label, 'batch_size': a.batch_size, 'epochs': a.epochs,
             'eval_nodes_per_split': a.eval_max_nodes, 'best_epoch': best['epoch'],
             'best_valid_acc_pct': round(100 * float(best['valid_acc']), 4),
             'test_at_best_valid_pct': round(100 * float(best['test_acc']), 4),
             'mean_training_s_per_epoch': round(mean_seconds, 3),
             'polled_peak_process_rss_mib': round(peak_rss / 1048576, 1),
             'total_run_wall_s': round(time.perf_counter() - start, 1),
             'per_epoch_csv': str(path)}
        results.append(r)
        summary = out / 'fanout_full_epoch_summary.csv'
        with summary.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(r))
            w.writeheader()
            w.writerows(results)
        print('DONE', r, flush=True)
    print('\nSummary:', out / 'fanout_full_epoch_summary.csv', flush=True)
    print('NOTE: RSS includes CPU dataset/runtime; sampled evaluation is NOT official OGB inference.', flush=True)

if __name__ == '__main__':
    main()
