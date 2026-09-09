"""Regenerate a run's loss plots from a saved checkpoint.

    python3 plot_checkpoint.py model_output_090826_128x128_unet/model_epoch499.pt

Checkpoints carry the whole `losses` list, so the end-of-training plots can be
produced for a run that was interrupted, or is still going, without retraining.
PNGs are written next to the checkpoint. An optional second argument names the
config to read batch_size from (default data_config.json).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")          # never open a window; this is a batch tool

import torch

from custom_data_io import DataConfig
from loss_plots import losses_to_series, plot_losses


def main(checkpoint_path, config_path=None):
    config = DataConfig.load(config_path) if config_path else DataConfig()
    output_dir = os.path.dirname(checkpoint_path) or "."

    ck = torch.load(checkpoint_path, map_location="cpu")
    losses = ck['loss']
    series = losses_to_series(losses, config.batch_size)
    epoch_nums, val, train, iter_nums, batch_losses = series

    print(f"{checkpoint_path}: epoch {ck['epoch']}, {len(losses)} loss entries")
    print(f"  {len(epoch_nums)} epochs of averages, {len(batch_losses)} gradient steps")
    if not epoch_nums:
        raise SystemExit("no per-epoch averages in this checkpoint -- nothing to plot")
    print(f"  final training loss   {train[-1]:.6e}")
    print(f"  final validation loss {val[-1]:.6e}")

    plot_losses(*series, output_dir)
    for name in ("loss_vs_epoch", "log_loss_vs_epoch", "loss_per_step", "log_loss_per_step"):
        print(f"  wrote {output_dir}/{name}.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(*sys.argv[1:3])
