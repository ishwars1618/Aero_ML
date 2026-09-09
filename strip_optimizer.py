"""Rewrite training checkpoints without their optimizer state.

    python3 strip_optimizer.py model_output_090826_128x128_unet
    python3 strip_optimizer.py model_output_*/ --all
    python3 strip_optimizer.py model_output_090826_128x128_unet --replace

save_checkpoint() in train_model.py stores the Adam state alongside the weights, and
Adam keeps two buffers (exp_avg, exp_avg_sq) per parameter -- so the optimizer state is
exactly twice the size of the model and about two thirds of every file:

    U-Net model_epoch499.pt   120.6 MiB total = 40.0 weights + 79.7 optimizer + 0.7 loss

That state only matters for RESUMING training. Inference does not touch it: video_demo's
load_model() reads model_state_dict alone. So archived checkpoints can drop it and shrink
~3x, which is the difference between clearing GitHub's 100 MiB per-file cap and not.
(gzip, for comparison, saves only ~9% -- trained float32 weights are near-incompressible.)

Written as `<name>_noopt.pt` next to the original. That suffix is deliberate: the resume
path, latest_checkpoint(), matches `model_epoch(\\d+).pt` with re.fullmatch, so a
_noopt file is invisible to it and can never be picked up as something to resume from.

By default the newest checkpoint in each directory -- the one a resumed run would
actually load -- is left alone, so stripping never costs you the ability to continue
training. Pass --all to strip it too. Originals are kept unless --replace is given, and
--replace re-reads each stripped file and checks the weights survived before unlinking.
"""

import argparse
import os
import re
import sys

import torch

CKPT_RE = re.compile(r"model_epoch(\d+)\.pt")
SUFFIX = "_noopt.pt"
DROP = "optimizer_state_dict"


def mib(path):
    return os.path.getsize(path) / 2**20


def candidates(target):
    """-> (list of checkpoint paths, path the resume logic would pick or None)."""
    if os.path.isfile(target):
        return [target], None
    names = sorted(os.listdir(target))
    paths = [os.path.join(target, n) for n in names
             if n.endswith(".pt") and not n.endswith(SUFFIX)]
    # Mirror latest_checkpoint() in train_model.py: highest epoch in the filename,
    # not newest mtime, and model_final.pt does not count.
    newest, best = None, -1
    for p in paths:
        m = CKPT_RE.fullmatch(os.path.basename(p))
        if m and int(m.group(1)) > best:
            newest, best = p, int(m.group(1))
    return paths, newest


def strip(path, replace):
    """Write path's checkpoint minus the optimizer state. -> (before, after) MiB."""
    out = path[: -len(".pt")] + SUFFIX
    ck = torch.load(path, map_location="cpu")
    if DROP not in ck:
        print(f"  {os.path.basename(path):32} already has no {DROP}, skipped")
        return None
    keys = [k for k in ck if k != DROP]
    torch.save({k: ck[k] for k in keys}, out)
    before, after = mib(path), mib(out)

    if replace:
        # Only unlink once the replacement has been read back and matches, so an
        # interrupted or truncated write can never lose the weights.
        check = torch.load(out, map_location="cpu")
        old_sd = ck["model_state_dict"]
        new_sd = check["model_state_dict"]
        assert new_sd.keys() == old_sd.keys(), f"{out}: state_dict keys changed"
        for k in old_sd:
            if torch.is_tensor(old_sd[k]):
                assert torch.equal(new_sd[k], old_sd[k]), f"{out}: {k} differs"
        os.remove(path)

    print(f"  {os.path.basename(path):32} {before:7.1f} -> {after:6.1f} MiB "
          f"({before / after:.1f}x)  -> {os.path.basename(out)}"
          f"{'  [original removed]' if replace else ''}")
    return before, after


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="checkpoint files or output directories")
    ap.add_argument("--all", action="store_true",
                    help="also strip the newest checkpoint (the resume point)")
    ap.add_argument("--replace", action="store_true",
                    help="delete each original after verifying its replacement")
    args = ap.parse_args()

    total_before = total_after = 0.0
    for target in args.targets:
        if not os.path.exists(target):
            sys.exit(f"no such file or directory: {target}")
        paths, newest = candidates(target)
        print(f"\n{target}")
        if not paths:
            print("  no checkpoints found")
            continue
        for p in paths:
            if p == newest and not args.all:
                print(f"  {os.path.basename(p):32} {mib(p):7.1f} MiB   "
                      f"KEPT -- this is the resume point (--all to strip it)")
                continue
            got = strip(p, args.replace)
            if got:
                total_before += got[0]
                total_after += got[1]

    if total_before:
        print(f"\ntotal {total_before:.1f} -> {total_after:.1f} MiB "
              f"(saved {total_before - total_after:.1f} MiB, "
              f"{total_before / total_after:.1f}x)")
        if not args.replace:
            print("originals kept; rerun with --replace to remove them once you are happy")


if __name__ == "__main__":
    main()
