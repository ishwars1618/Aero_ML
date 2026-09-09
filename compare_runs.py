"""Compare training/validation loss curves across runs.

    python3 compare_runs.py out.png labelA=pathA.pt labelB=pathB.pt ...

Reads the `losses` list each checkpoint carries, so runs can be compared after the
fact without retraining. One colour per run; training solid, validation dashed.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from custom_data_io import DataConfig
from loss_plots import losses_to_series


def series(path, batch_size):
    ck = torch.load(path, map_location="cpu")
    epochs, val, train, _, _ = losses_to_series(ck['loss'], batch_size)
    return np.array(epochs), np.array(train), np.array(val)


def main(out_path, *specs):
    # batch_size scales every curve by the same constant, so the comparison does
    # not depend on which config a run used; take the dataclass default.
    batch_size = DataConfig().batch_size
    runs = [s.split("=", 1) for s in specs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 0.3, len(runs)))
    print(f"{'run':28} {'epochs':>7} {'final train':>13} {'final val':>13} {'val/train':>10} {'best val':>13} {'@epoch':>7}")
    for (label, path), c in zip(runs, colors):
        ep, tr, va = series(path, batch_size)
        for ax, logy in zip(axes, (False, True)):
            y_tr, y_va = (np.log10(tr), np.log10(va)) if logy else (tr, va)
            # carry each curve's final value in its legend entry, formatted for the
            # panel it is on: linear panels in scientific notation, log panels as
            # the log10 value actually plotted.
            fmt = (lambda y: f"{y:+.2f}") if logy else (lambda y: f"{y:.3e}")
            ax.plot(ep, y_tr, color=c, linewidth=1.2,
                    label=f"{label} train  (final {fmt(y_tr[-1])})")
            ax.plot(ep, y_va, color=c, linewidth=1.2, linestyle="--", alpha=0.85,
                    label=f"{label} val    (final {fmt(y_va[-1])})")
        b = int(np.argmin(va))
        print(f"{label:28} {len(ep):7d} {tr[-1]:13.4e} {va[-1]:13.4e} {va[-1]/tr[-1]:10.2f} {va[b]:13.4e} {ep[b]:7d}")

    axes[0].set_title("Training vs validation loss (softmax->MSE)")
    axes[0].set_ylabel("MSE loss"); axes[0].set_ylim(bottom=0)
    axes[1].set_title("Log training vs validation loss (softmax->MSE)")
    axes[1].set_ylabel("log10 MSE loss")
    for ax in axes:
        ax.set_xlabel("Epochs"); ax.grid(alpha=0.25)
        ax.legend(fontsize=8, prop={"family": "monospace", "size": 8})
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], *sys.argv[2:])
