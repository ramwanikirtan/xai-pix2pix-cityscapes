"""Plot training loss curves from the loss_log.txt file.

Produces ./logs/phase1_loss_curve.png for the paper (Figure 2).

Usage:
    python scripts/plot_loss_curve.py --log_file checkpoints/cityscapes_pix2pix/loss_log.txt
"""
import argparse
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_loss_log(log_file):
    """Parse the loss_log.txt file written by the visualizer."""
    losses = {}
    epochs = []

    with open(log_file, 'r') as f:
        for line in f:
            # Format: [Rank 0] (epoch: X, iters: Y, time: Z, data: W) , G_GAN: 1.234 ...
            match = re.search(r'\(epoch:\s*(\d+),\s*iters:\s*(\d+)', line)
            if match:
                epoch = int(match.group(1))
                # Parse loss values
                loss_pairs = re.findall(r'(\w+):\s*([\d.e+-]+)', line)
                for name, val in loss_pairs:
                    if name in ('epoch', 'iters', 'time', 'data'):
                        continue
                    if name not in losses:
                        losses[name] = []
                    losses[name].append((epoch, float(val)))

    return losses


def plot_losses(losses, output_path):
    """Plot all losses and save to file."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Generator losses
    ax_g = axes[0]
    for name in ['G_GAN', 'G_L1']:
        if name in losses:
            data = losses[name]
            epochs = [d[0] for d in data]
            values = [d[1] for d in data]
            # Smooth with moving average
            window = min(50, len(values) // 10) if len(values) > 100 else 1
            if window > 1:
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                ax_g.plot(range(len(smoothed)), smoothed, label=name, alpha=0.9)
            else:
                ax_g.plot(values, label=name, alpha=0.9)

    ax_g.set_xlabel('Iteration')
    ax_g.set_ylabel('Loss')
    ax_g.set_title('Generator Losses')
    ax_g.legend()
    ax_g.grid(True, alpha=0.3)

    # Plot Discriminator losses
    ax_d = axes[1]
    for name in ['D_real', 'D_fake']:
        if name in losses:
            data = losses[name]
            values = [d[1] for d in data]
            window = min(50, len(values) // 10) if len(values) > 100 else 1
            if window > 1:
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                ax_d.plot(range(len(smoothed)), smoothed, label=name, alpha=0.9)
            else:
                ax_d.plot(values, label=name, alpha=0.9)

    ax_d.set_xlabel('Iteration')
    ax_d.set_ylabel('Loss')
    ax_d.set_title('Discriminator Losses')
    ax_d.legend()
    ax_d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str,
                        default='checkpoints/cityscapes_pix2pix/loss_log.txt',
                        help='Path to loss_log.txt')
    parser.add_argument('--output', type=str,
                        default='./logs/phase1_loss_curve.png',
                        help='Output PNG path')
    args = parser.parse_args()

    Path(args.output).parent.mkdir(exist_ok=True)

    losses = parse_loss_log(args.log_file)
    if not losses:
        print(f"No losses found in {args.log_file}")
        return

    print(f"Found losses: {list(losses.keys())}")
    for name, data in losses.items():
        print(f"  {name}: {len(data)} entries, last value: {data[-1][1]:.4f}")

    plot_losses(losses, args.output)


if __name__ == '__main__':
    main()
