# -*- coding: utf-8 -*-
"""
绘制 Option 2 训练日志

用法示例：
    python plot_results.py --logs model/5_5_4_block1/training_log.csv model/5_5_4_block2/training_log.csv --labels block1 block2 --output result.png
"""

import argparse
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot(logs, labels, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for path, label in zip(logs, labels):
        if not os.path.exists(path):
            print(f'Warning: {path} not found, skip.')
            continue
        data = load_csv(path)
        batches = [int(r['batch']) for r in data]
        losses = [float(r['loss']) if r['loss'] else None for r in data]
        entropies = [float(r['entropy']) if r['entropy'] else None for r in data]
        kls = [float(r['kl']) if r['kl'] else None for r in data]
        win_ratios = [float(r['win_ratio']) if r['win_ratio'] else None for r in data]

        axes[0, 0].plot(batches, losses, label=label)
        axes[0, 1].plot(batches, entropies, label=label)
        axes[1, 0].plot(batches, kls, label=label)
        axes[1, 1].plot(batches, win_ratios, label=label, marker='o')

    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('batch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].set_title('Entropy')
    axes[0, 1].set_xlabel('batch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].set_title('KL')
    axes[1, 0].set_xlabel('batch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].set_title('Win Ratio vs Pure MCTS')
    axes[1, 1].set_xlabel('batch')
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print('Saved plot to', output)


def parse_args():
    parser = argparse.ArgumentParser(description='绘制训练曲线')
    parser.add_argument('--logs', type=str, nargs='+', required=True,
                        help='一个或多个 training_log.csv 路径')
    parser.add_argument('--labels', type=str, nargs='+', required=True,
                        help='与 logs 一一对应的图例标签')
    parser.add_argument('--output', type=str, default='result.png',
                        help='输出图片路径')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if len(args.logs) != len(args.labels):
        raise ValueError('--logs 与 --labels 数量必须相同')
    plot(args.logs, args.labels, args.output)
