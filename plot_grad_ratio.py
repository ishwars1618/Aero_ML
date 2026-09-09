"""Plot the gradient-to-value ratio per module, parsed out of a run's training.log.

    python3 plot_grad_ratio.py model_output_090826_128x128_unet

log_module_stats writes a "Gradient statistics" block and a "Value statistics" block
per training batch, each one line per module: `name: min to median to max`. This
reads both, divides the medians, and plots median|grad| / median|value| over time.

That ratio, rather than the raw gradient magnitude, is what says whether a layer is
learning: a gradient of 1e-4 is enormous for weights of 1e-5 and negligible for
weights of 1. Roughly 1e-3 to 1e-2 per step is healthy; a line drifting toward zero
is a layer going dead, and one spiking well above the rest is where instability
starts.
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STAT_RE = re.compile(r"^\t([\w.]+): ([\d.e+-]+) to ([\d.e+-]+) to ([\d.e+-]+)$")


def parse_log(path):
    """-> (module_names, grads, values), each array (n_batches, n_modules) of medians."""
    grads, values = [], []
    section, cur_g, cur_v = None, {}, {}
    order = []
    with open(path) as f:
        for line in f:
            if line.startswith("\tGradient statistics"):
                # a complete pair from the previous batch is ready to bank
                if cur_g and cur_v:
                    grads.append(cur_g); values.append(cur_v)
                section, cur_g, cur_v = "grad", {}, {}
                continue
            if line.startswith("\tValue statistics"):
                section = "value"
                continue
            m = STAT_RE.match(line.rstrip("\n"))
            if not m or section is None:
                continue
            name, median = m.group(1), float(m.group(3))
            if name not in order:
                order.append(name)
            (cur_g if section == "grad" else cur_v)[name] = median
    if cur_g and cur_v:
        grads.append(cur_g); values.append(cur_v)

    g = np.array([[b.get(n, np.nan) for n in order] for b in grads])
    v = np.array([[b.get(n, np.nan) for n in values[0]] for b in values]) if values else np.empty((0, 0))
    v = np.array([[b.get(n, np.nan) for n in order] for b in values])
    return order, g, v


def bucket(a, n_buckets):
    """Median within each of n_buckets consecutive slices, to keep the plot legible."""
    idx = np.array_split(np.arange(len(a)), n_buckets)
    return np.array([np.nanmedian(a[i], axis=0) for i in idx if len(i)])


def main(output_dir):
    names, g, v = parse_log(os.path.join(output_dir, "training.log"))
    print(f"parsed {g.shape[0]} batches x {g.shape[1]} modules")
    ratio = g / v                                     # median|grad| / median|value|

    n_buckets = min(500, len(ratio))
    r = bucket(ratio, n_buckets)
    x = np.linspace(0, len(ratio), len(r))

    plt.figure(figsize=(11, 6.5))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(names)))
    for i, name in enumerate(names):
        plt.plot(x, r[:, i], label=name, color=colors[i], linewidth=1.1)
    plt.yscale("log")
    plt.title("Gradient / value ratio per module (median |grad| / median |weight|)")
    plt.xlabel("Training batch")
    plt.ylabel("ratio")
    plt.axhspan(1e-3, 1e-2, color="grey", alpha=0.15, zorder=0)
    plt.legend(fontsize=7, ncol=2, loc="upper right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = os.path.join(output_dir, "grad_value_ratio.png")
    plt.savefig(out, dpi=130)
    plt.clf()
    print(f"wrote {out}   (shaded band = the usual healthy 1e-3..1e-2)")

    print("\nfinal-decile median ratio, in model order:")
    tail = np.nanmedian(ratio[-len(ratio)//10:], axis=0)
    for name, val in zip(names, tail):
        print(f"  {name:24} {val:.2e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
