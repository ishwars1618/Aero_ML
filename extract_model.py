"""Extract definitions from notebooks into importable .py modules.

An .ipynb is just JSON, so this reads it with the stdlib `json` module and picks
out the cells worth keeping.

Cells are selected by *reachability*: name the public roots, and `ast` walks the
notebook's top-level definitions to find everything they transitively depend on.
Nothing is addressed by cell index, so reordering or inserting notebook cells
does not break the extraction, and dead experimental code is left behind
automatically.

    python3 extract_model.py                # regenerate every module
    python3 extract_model.py vit            # just one
    python3 extract_model.py --check        # exit 1 if any is stale

One escape hatch, index-free:

  extra    names to include even though the roots never reach them; emitted
           first, right after the imports

The data pipeline used to be generated here too. It is now hand-maintained in
custom_data_io.py, because its classes take explicit config arguments instead of
closing over notebook globals -- so it is no longer a copy of the notebook text.
The config/wrap/mirror machinery that supported it was removed with it.
"""

import argparse
import ast
import builtins
import json
import re
import sys

UNET_NB = "unet_with_supervision_stacked.ipynb"
VIT_NB = "vit3.ipynb"

TARGETS = {
    "unet": {
        "notebook": UNET_NB,
        "output": "unet_model.py",
        "roots": ["StackedHourglass"],
        "imports": ["import torch", "import torch.nn as nn",
                    "import torch.nn.functional as F"],
        "doc": "U-Net / stacked-hourglass landmark model.",
    },
    "vit": {
        "notebook": VIT_NB,
        "output": "vit_model.py",
        "roots": ["VisionTransformer", "weights_init"],
        # Not reachable from the roots (the notebook only references these from
        # commented-out code), but kept for experimentation.
        "extra": ["SameSizeConv", "gen_ape", "VisionAttention"],
        "imports": ["import torch", "import torch.nn as nn",
                    "import torch.nn.functional as F"],
        "doc": "Vision-transformer landmark model.",
    },
    "vit_script": {
        # Everything in the ViT notebook that did NOT become one of the extracted
        # modules: setup, training loop, plots, attention probes, video demo.
        "notebook": VIT_NB,
        "output": "vit3.py",
        "mode": "script",
        "exclude_generated": ["vit_model.py"],
        # Cells that became custom_data_io.py. They come from this notebook, so
        # they are named by index rather than matched against another file.
        "exclude_cells": [15, 16, 17, 18, 19, 26, 27, 28],
        # Cells whose text is already in video_demo.py, matched by content
        # against the U-Net notebook cells that file was cut from. The ViT's own
        # run_video_demo_auto (cell 66) differs -- it wraps inference in
        # autocast -- so it does not match and stays here.
        "exclude_like": (UNET_NB, [45, 46, 47, 48, 49]),
        "preamble": [
            "# Names the excluded cells used to provide, now imported from the",
            "# extracted modules. `threading` and the frame_manager helpers are",
            "# imported here because the video-demo cells below use them",
            "# unqualified but the notebook never imports them.",
            "import threading",
            "",
            "from frame_manager import (draw_points, heatmaps_to_coords,",
            "                           preprocess_frame, put_fps)",
            "from vit_model import (VisionTransformer, weights_init, SameSizeConv,",
            "                       gen_ape, VisionAttention, MultiHeadVisionAttention,",
            "                       ChannelLayerNorm, ResidualNorm, Residual, MLP,",
            "                       PatchConv, gen_learnable_ape)",
            "from custom_data_io import (DataConfig, FramesFromCSV, ComposeJoint,",
            "                            RandomAffineWithPoints, RandomColorMatrix,",
            "                            RandomAdditiveNoise, coord_to_heatmap,",
            "                            gen_coordinates, build_dataset,",
            "                            build_dataloaders)",
            "",
            "config = DataConfig.load()",
            "dataset = build_dataset(config)",
            "train_loader, val_loader, test_loader = build_dataloaders(dataset, config)",
            "train_size, val_size = len(train_loader.dataset), len(val_loader.dataset)",
        ],
        "doc": "Training, evaluation and demo script for the vision transformer.",
    },
    "unet_script": {
        # Everything in the U-Net notebook that did NOT become one of the
        # extracted modules: setup, training loop, plots, checkpointing, the
        # video demo. A linear transcript, not a library.
        "notebook": UNET_NB,
        "output": "train_model.py",
        "mode": "script",
        # Cells already emitted verbatim into these generated modules; read back
        # out of their own provenance comments.
        "exclude_generated": ["unet_model.py"],
        # Cells that became custom_data_io.py. That file is hand-maintained and
        # no longer carries provenance, so identify them by matching content
        # against the ViT notebook cells they were taken from.
        "exclude_like": (VIT_NB, [15, 16, 17, 18, 19, 26, 27, 28]),
        # Cells lifted verbatim into video_demo.py (hand-maintained), by index:
        # they come from this notebook, so there is no other file to match against.
        "exclude_cells": [45, 46, 47, 48, 49],
        "preamble": [
            "# Names the excluded cells used to provide, now imported from the",
            "# extracted modules.",
            "",
            "from unet_model import (SameSizeConv, Up, Up2, Down, UNet,",
            "                        StackedHourglass)",
            "from custom_data_io import (DataConfig, FramesFromCSV, ComposeJoint,",
            "                            RandomAffineWithPoints, RandomColorMatrix,",
            "                            RandomAdditiveNoise, coord_to_heatmap,",
            "                            gen_coordinates, build_dataset,",
            "                            build_dataloaders)",
            "",
            "config = DataConfig.load()",
            "dataset = build_dataset(config)",
            "train_loader, val_loader, test_loader = build_dataloaders(dataset, config)",
            "train_size, val_size = len(train_loader.dataset), len(val_loader.dataset)",
        ],
        "doc": "Training, evaluation and demo script for the stacked-hourglass U-Net.",
    },
}

SCRIPT_HEADER = '''"""{doc}

AUTOGENERATED by extract_model.py from {notebook} -- do not edit by hand.
Regenerate with:  python3 extract_model.py {name}

A linear transcript of every notebook cell that did not become an importable
module, in notebook order, demarcated with `# --- cell N`. Running this file
runs all of it -- training loop, plots and video demo included -- so comment out
what you do not want before invoking it.
"""

{preamble}

'''

HEADER = '''"""{doc}

AUTOGENERATED by extract_model.py from {notebook} -- do not edit by hand.
Regenerate with:  python3 extract_model.py {name}

Public API: {api}
"""

{imports}

'''


def code_cells(nb):
    """Yield (index, source) for every code cell.

    `source` in nbformat 4 is a list of lines (each keeping its trailing "\\n"),
    but the spec also permits a single string, so normalize both.
    """
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        yield i, src if isinstance(src, str) else "".join(src)


def parse(src):
    """Parse a cell, or None if it is not valid Python.

    Cells using IPython magics (%matplotlib, !pip) are skipped rather than
    treated as an error.
    """
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


def referenced_names(node):
    """Every bare name the subtree reads, plus roots of attribute chains.

    `nn.Module` -> nn, `torch.cat` -> torch. Also picks up `global X` so a
    forward() that mutates module-level state keeps that state.
    """
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            names.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            names.add(n.value.id)
        elif isinstance(n, ast.Global):
            names.update(n.names)
    return names


def index(nb):
    """Map notebook top-level names to the cell that binds them."""
    cells, def_home, deps = {}, {}, {}
    for idx, src in code_cells(nb):
        tree = parse(src)
        if tree is None:
            continue
        cells[idx] = src
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                # Later redefinitions win, matching top-to-bottom execution.
                def_home[node.name] = idx
                deps[node.name] = referenced_names(node)
    return cells, def_home, deps


def build_script(spec, name):
    """Everything the module extractions left behind, in notebook order."""
    with open(spec["notebook"]) as f:
        nb = json.load(f)
    cells = {i: ("".join(c["source"]) if not isinstance(c.get("source"), str)
                 else c["source"], c.get("cell_type"))
             for i, c in enumerate(nb["cells"])}

    excluded = set(spec.get("exclude_cells", []))
    for path in spec.get("exclude_generated", []):
        excluded |= {int(n) for n in re.findall(
            r"# --- from notebook cell (\d+) ---", open(path).read())}
    mirror_nb, idxs = spec.get("exclude_like", (None, []))
    if mirror_nb:
        with open(mirror_nb) as f:
            mv = json.load(f)
        srcs = {"".join(mv["cells"][i]["source"]) for i in idxs}
        excluded |= {i for i, (src, _) in cells.items() if src in srcs}

    chunks = []
    for i in sorted(cells):
        if i in excluded:
            continue
        src, kind = cells[i]
        if kind != "code":
            # Keep markdown as a comment so cell numbering stays aligned.
            body = "\n".join(f"# {ln}" for ln in src.rstrip().splitlines())
            chunks.append(f"# --- cell {i} ({kind})\n{body}\n")
        else:
            chunks.append(f"# --- cell {i}\n{src.rstrip()}\n")

    kept = sorted(set(cells) - excluded)
    print(f"note: {spec['output']}: kept {len(kept)}/{len(cells)} cells; "
          f"excluded {sorted(excluded)}", file=sys.stderr)
    return SCRIPT_HEADER.format(doc=spec["doc"], notebook=spec["notebook"],
                                name=name,
                                preamble="\n".join(spec["preamble"])) + \
        "\n\n".join(chunks) + "\n"


def build(spec, name):
    if spec.get("mode") == "script":
        return build_script(spec, name)
    with open(spec["notebook"]) as f:
        nb = json.load(f)
    cells, def_home, deps = index(nb)

    roots, extra = list(spec["roots"]), list(spec.get("extra", []))
    missing = [r for r in roots + extra if r not in def_home]
    if missing:
        raise SystemExit(f"{spec['notebook']}: not found: {', '.join(missing)}")

    def closure(seeds):
        """Every notebook-defined name transitively reachable from `seeds`."""
        found, queue = set(), list(seeds)
        while queue:
            n = queue.pop()
            if n in found or n not in def_home:
                continue
            found.add(n)
            queue.extend(deps.get(n, ()))
        return found

    needed = closure(roots)
    extra_needed = closure(extra) - needed

    # Emit whole cells, in notebook order. A notebook that runs top-to-bottom is
    # already in define-before-use order, so no topological sort is needed.
    #
    # Extras go first, immediately after the imports. That is safe even when an
    # extra references a class defined further down (VisionAttention uses
    # ResidualNorm and MLP): a class body does not evaluate its method bodies at
    # definition time, so the name only has to exist by the time it is called.
    core_cells = sorted({def_home[n] for n in needed})
    extra_cells = sorted({def_home[n] for n in extra_needed} - set(core_cells))

    chunks = [f"# --- from notebook cell {idx} ---\n{cells[idx].rstrip()}\n"
              for idx in extra_cells + core_cells]
    body = "\n\n".join(chunks)

    # Report anything the extracted code still cannot resolve.
    tree = ast.parse(body)
    bound = {n.name for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    bound |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    bound |= {a.arg for n in ast.walk(tree) if isinstance(n, ast.arguments)
              for a in n.args + n.posonlyargs + n.kwonlyargs}
    provided = {ast.unparse(a).split()[-1] for imp in spec["imports"]
                for a in ast.parse(imp).body[0].names}
    provided |= {a.asname or a.name.split(".")[0] for imp in spec["imports"]
                 for a in ast.parse(imp).body[0].names}
    free = referenced_names(tree) - bound - set(dir(builtins)) - provided
    if free:
        print(f"warning: {spec['output']}: unresolved names {sorted(free)}",
              file=sys.stderr)

    # Definitions the notebook has but the roots never reach: left out on purpose.
    unreachable = sorted(set(def_home) - needed - extra_needed)
    if unreachable:
        print(f"note: {spec['output']}: not reachable from roots, omitted: "
              f"{', '.join(unreachable)}", file=sys.stderr)

    api = ", ".join(roots + extra)
    header = HEADER.format(doc=spec["doc"], notebook=spec["notebook"], name=name,
                           api=api, imports="\n".join(spec["imports"]))
    return header + body + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", choices=list(TARGETS) + [[]],
                    help="which modules to extract (default: all)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any generated file is out of date")
    args = ap.parse_args()

    stale = 0
    for name in (args.targets or TARGETS):
        spec = TARGETS[name]
        text = build(spec, name)
        if args.check:
            try:
                current = open(spec["output"]).read()
            except FileNotFoundError:
                current = None
            if current != text:
                print(f"{spec['output']} is out of date; run: "
                      f"python3 extract_model.py {name}", file=sys.stderr)
                stale = 1
            else:
                print(f"{spec['output']} is up to date")
        else:
            with open(spec["output"], "w") as f:
                f.write(text)
            print(f"wrote {spec['output']} ({len(text.splitlines())} lines)")
    return stale


if __name__ == "__main__":
    sys.exit(main())
