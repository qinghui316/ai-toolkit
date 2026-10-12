"""Reproduce diagnostic curves from authoritative per-update CSV records."""
import argparse
import csv
from pathlib import Path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('steps_csv', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with args.steps_csv.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    x = [int(row['completed_updates']) for row in rows]
    for column, label in [('loss/base', 'Base'), ('loss/cnn_weighted', 'Weighted CNN'), ('loss/total', 'Total')]:
        if rows and column in rows[0]:
            axes[0].plot(x, [float(row[column]) for row in rows], label=label)
    axes[0].set_ylabel('Engineering loss')
    axes[0].legend()
    axes[1].plot(x, [float(row['cuda_peak_allocated']) / 2**30 for row in rows], label='Peak allocated')
    axes[1].plot(x, [float(row['cuda_peak_reserved']) / 2**30 for row in rows], label='Peak reserved')
    axes[1].set_ylabel('GPU memory (GiB)')
    axes[1].set_xlabel('Successful optimizer updates')
    axes[1].legend()
    fig.suptitle('Engineering smoke test — not a generative quality benchmark')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(args.output.with_suffix('.' + suffix), dpi=160)
