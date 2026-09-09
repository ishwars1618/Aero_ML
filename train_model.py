from unet_model import (SameSizeConv, Up, Up2, Down, UNet,
                        StackedHourglass)
from model_registry import build_model
from loss_plots import losses_to_series, plot_losses
from custom_data_io import (DataConfig, FramesFromCSV, ComposeJoint,
                            RandomAffineWithPoints, RandomColorMatrix,
                            RandomAdditiveNoise, coord_to_heatmap,
                            gen_coordinates, build_dataset,
                            build_dataloaders)

# Which config file to run: `python train_model.py vit_config.json`,
# otherwise the default data_config.json. Keeping lr/clip/output_dir/model in one
# file means a run is described entirely by its config.
import sys
config = DataConfig.load(sys.argv[1]) if len(sys.argv) > 1 else DataConfig.load()
dataset = build_dataset(config)
train_loader, val_loader, test_loader = build_dataloaders(dataset, config)
train_size, val_size = len(train_loader.dataset), len(val_loader.dataset)

# --- cell 0

#%matplotlib inline
import argparse
import os
import random
import re
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from PIL import Image
import skimage
from skimage import io, transform


# --- cell 1
# All of these now come from data_config.json, so a run is described by the config
# rather than by edits to this file.
batch_size = config.batch_size
image_size = config.image_size
model_image_size = config.model_image_size
workers = config.workers
rootdir = config.rootdir
output_dir = config.output_dir
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
NUM_LANDMARKS = config.num_landmarks


# --- cell 8
# Dont USE
weight_spread = 0.05
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        if hasattr(m.bias, 'data'):
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.05)
        nn.init.normal_(m.bias.data, 0.0, 0.001)
    elif classname.find('Conv2d') != -1:
        # If 'Conv' is found, apply a specific initialization
        #nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
        #nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.weight.data, 0.0, 1.5*(2/(m.weight.shape[0]*m.weight.shape[2]*m.weight.shape[3]))**0.5)
        print(m.weight.shape)
        if hasattr(m.bias, 'data'):
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('ConvTranspose2d') != -1:
        # If 'Conv' is found, apply a specific initialization
        #nn.init.normal_(m.weight.data, 0.0, 0.01)
        #nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
        nn.init.normal_(m.weight.data, 0.0, (2/(m.weight.shape[0]*2*2))**0.5)
        print(m.weight.shape)
        if hasattr(m.bias, 'data'):
            nn.init.constant_(m.bias.data, 0.0)
    m = m.to(torch.float32)

clip_value =  5
class WeightConstraint(object):
    def __init__(self):
        pass
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-clip_value, clip_value)
            module.weight.data = w


# --- cell 9
# Set random seed for reproducibility
manualSeed = 499
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

torch.use_deterministic_algorithms(True) # Needed for reproducible results

def test_dataset(dataset, dir=None):
    if dir is None:
        dir = f"{output_dir}/test_dataset"

    log_path = f"{dir}/test_dataset.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "a")

    # --- cell 15
    IMG_IDX = 291
    IMG = dataset[IMG_IDX]
    plt.imshow(torch.permute(IMG['image'], (1, 2, 0))*0.5 + 0.5)
    # Unpack points into two lists: xs, ys
    xs = [p[0] for p in IMG['points']]
    ys = [p[1] for p in IMG['points']]
    # Plot points as red circles
    plt.scatter(xs, ys, c='red', s=10)  # s = point size
    plt.savefig(f"{dir}/i{IMG_IDX}_image_points.png")
    plt.clf()


    # --- cell 16
    f.write(f"{torch.min(dataset[IMG_IDX]['image'])}\n")
    f.write(f"{torch.max(dataset[IMG_IDX]['image'])}\n")
    f.write(f"{dataset[IMG_IDX]['image'].shape}\n")


    # --- cell 17
    f.write(f"{torch.min(IMG['heatmaps'])} {torch.max(IMG['heatmaps'])}\n")
    f.write(f"{IMG['heatmaps'].shape}\n")
    plt.imshow(IMG['heatmaps'][1])
    plt.colorbar()
    plt.savefig(f"{dir}/i{IMG_IDX}_heatmap1.png")
    plt.clf()


    # --- cell 18
    f.write(f"{torch.mean(IMG['heatmaps'][1])}\n")


    # --- cell 19
    f.write(f"{IMG['points']}\n")


    # --- cell 20
    f.write(f"{torch.min(IMG['heatmaps'])} {torch.max(IMG['heatmaps'])}\n")
    plt.imshow(torch.sum(IMG['heatmaps'], axis=0))
    plt.colorbar()
    plt.savefig(f"{dir}/i{IMG_IDX}_heatmaps_sum.png")
    plt.clf()

    f.close()


def _abs_stats(submodule, which, recurse):
    """|grad| or |data| of a submodule's parameters as (min, median, max), or None."""
    vals = []
    for _, param in submodule.named_parameters(recurse=recurse):
        tensor = param.grad if which == "grad" else param.data
        if tensor is not None:
            vals.append(tensor.detach().view(-1).cpu().numpy())
    if not vals:
        return None
    vals = np.abs(np.concatenate(vals))
    return np.min(vals), np.median(vals), np.max(vals)


def module_param_stats(module, which="grad", groups_only=True):
    """Summarize |grad| or |value| of the parameters under `module`.

    Yields (name, min, median, max). Which submodules get a line is declared in the
    model definition, by a `log_stats = True` attribute (see Down/Up/Up2/SameSizeConv
    in unet_model.py): a tagged submodule is reported as the aggregate over its whole
    subtree and is not descended into. Anything not covered by a tagged subtree is
    reported at the level that registered it, so every parameter appears exactly once
    no matter how the tags are placed.

    groups_only=False ignores the tags and reports every submodule individually.
    Submodules with no parameters -- or, for which="grad", no .grad yet -- are skipped.
    """
    def walk(submodule, name):
        if groups_only and getattr(submodule, "log_stats", False):
            stats = _abs_stats(submodule, which, recurse=True)
            if stats is not None:
                yield (name, *stats)
            return                      # tagged: aggregate here, don't descend
        stats = _abs_stats(submodule, which, recurse=False)
        if stats is not None:
            yield (name or type(module).__name__, *stats)
        for child_name, child in submodule.named_children():
            yield from walk(child, f"{name}.{child_name}" if name else child_name)

    yield from walk(module, "")


def log_module_stats(module, f_log, groups_only=True, indent="\t"):
    for which, header in (("grad", "Gradient statistics"), ("value", "Value statistics")):
        f_log.write(f"{indent}{header}\n")
        for name, lo, med, hi in module_param_stats(module, which, groups_only):
            f_log.write(f"{indent}{name}: {lo:.2e} to {med:.2e} to {hi:.2e}\n")


# --- cell 24
trial_number = 0

from datetime import datetime
import time

device = torch.device("mps")
print("device: ", device)

print(f"Instantiating model {config.model!r}")
model = build_model(config).to(device, dtype=torch.float32)
# Weight init is the registry's job -- it differs per architecture and is required
# for the vit. The weights_init defined above is the U-Net's disabled experiment.
print(model)

# --- cell 37
# Assuming 'model' is your defined torch.nn.Module instance
pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {pytorch_total_params}")
print(f"Trainable parameters: {pytorch_trainable_params}")

# Setup Adam optimizers for both G and D
optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.5, 0.999))
print("Model optimizer: \n", optimizer)

criterion = nn.MSELoss()

# --- cell 25
iters = 0
epoch = 0
num_epochs = config.num_epochs
#k=500 # saving images of fixed_noise outputs
last_save = iters
break_flag = False

losses = []

example_idx = 110


# --- cell 26
# heatmap_softmax now lives on StackedHourglass, alongside error(); see unet_model.py.
LOSS_SCALE = (10**2)**3   # factor to adjust for the scale of the heatmaps

# --- cell 27

def training_step(model, data, epoch, i, f_log, mode="train"):
    """Run one batch.

    mode="train" -- model in .train() mode, grad enabled: zero_grad, loss, backward,
                    gradient stats, clipping, optimizer.step, and iters advances.
    mode="eval"  -- model in .eval() mode, grad disabled: forward and loss only, the
                    weights and iters are left untouched.

    Returns err_sum.item() for the caller's running total.
    """
    global iters

    train = (mode == "train")
    model.train() if train else model.eval()

    with torch.set_grad_enabled(train):
        if train:
            print(f"Epoch\t{epoch}, batch\t{i}.")
        t0 = time.perf_counter()

        data_input = data['image'].to(device, dtype=torch.float32)
        output = data['heatmaps'].to(device, dtype=torch.float32)
        output_coords = data['points'] / model_image_size

        b = data_input.shape[0]
        c = output.shape[1]

        t1 = time.perf_counter()
        if train:
            f_log.write(f"Epoch\t{epoch}, batch\t{i}, {b} elements.\n")
        else:
            f_log.write(f"Batch\t{i}, {b} elements.\n")
        f_log.write(f"\t\tTime for setup: {t1-t0}\n")

        heatmap_logits = model(data_input)
        expected_output_stages = output #output.repeat(1, 3, 1, 1)

        t2 = time.perf_counter()
        f_log.write(f"\t\tTime for forward: {t2-t1}\n")

        t3 = t2
        if train:
            model.zero_grad()

            t3 = time.perf_counter()
            f_log.write(f"\t\tTime for zero-grad: {t3-t2}\n")

        #print(expected_output_stages.shape)
        #print(predicted_heatmap_stages.shape)
        err_sum = model.error(heatmap_logits, expected_output_stages, criterion,
                              batch_size=batch_size, loss_scale=LOSS_SCALE)

        if train:
            losses.append( (epoch, i, iters, {f"Stage 0 error": err_sum.item()} ) )

            f_log.write(f"\tTraining loss / batch: {err_sum.item()}\n")
        else:
            f_log.write(f"\tValidation loss / batch: {err_sum.item()}\n")

        t4 = time.perf_counter()
        if train:
            f_log.write(f"\t\tTime for loss computation: {t4-t3}\n")
        else:
            f_log.write(f"\t\tTime for validation loss computation: {t4-t2}\n")

        if train:
            err_sum.backward()

            t5 = time.perf_counter()
            f_log.write(f"\t\tTime for backprop: {t5-t4}\n")

            log_module_stats(model, f_log)

            t6 = time.perf_counter()
            f_log.write(f"\t\tTime for gathering statistics: {t6-t5}\n")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_max_norm)

            t7 = time.perf_counter()
            f_log.write(f"\t\tTime for grad clip: {t7-t6}\n")

            optimizer.step()

            t8 = time.perf_counter()
            f_log.write(f"\t\tTime for opt step: {t8-t7}\n")

            iters += 1

        f_log.flush()

    return err_sum.item()


def save_example_illustration(model, dataset, example_idx, epoch, f_log):
    """Write the input image, the true heatmaps, and the model's logit and
    probability heatmaps for one dataset example, as PNGs under output_dir."""
    print(f"Saving results for example index {example_idx}")
    f_log.write(f"Saving results for example index {example_idx}\n")
    # output example output from model
    example_point = dataset[example_idx]
    image = torch.permute(example_point['image'], (1, 2, 0))
    plt.imshow(image.numpy())
    plt.savefig(f"{output_dir}/epoch{epoch}_i{example_idx}_in.png")
    plt.clf()

    input_example = example_point['image'].to(device=device, dtype=torch.float32)
    print(torch.min(input_example), torch.max(input_example))
    print(input_example.shape)

    heatmap_pred_detach, heatmap_probs_detach = model.forward_display( torch.stack((input_example,)) )

    heatmap_actual = example_point['heatmaps']
    print(heatmap_pred_detach[0].shape)
    for i in range(NUM_LANDMARKS):
        plt.imshow(heatmap_actual.numpy()[i])
        plt.colorbar()
        plt.savefig(f"{output_dir}/epoch{epoch}_i{example_idx}_out_{i}_real.png")
        plt.clf()
        for j in range(len(heatmap_pred_detach)):
            plt.title(f"Heatmap {i} logits")
            plt.imshow(heatmap_pred_detach[j].numpy()[0][i])
            plt.colorbar()
            plt.savefig(f"{output_dir}/epoch{epoch}_i{example_idx}_out_{i}_endpt{j}_logits.png")
            plt.clf()
            plt.title(f"Heatmap {i} probabilities")
            plt.imshow(heatmap_probs_detach[j].numpy()[0][i])
            plt.colorbar()
            plt.savefig(f"{output_dir}/epoch{epoch}_i{example_idx}_out_{i}_endpt{j}_probs.png")
            plt.clf()
        # for j in range(len(heatmap_pred_detach[1])):
        #     plt.imshow(heatmap_pred_detach[1][j].numpy()[0][i])
        #     plt.colorbar()
        #     plt.savefig(f"{output_dir}/epoch{epoch}_out_{i}_inter{j}.png")
        #     plt.clf()


def save_checkpoint(model, optimizer, epoch, losses, iters, path):
    """Write a resumable checkpoint: same payload as the final save at the end of
    the script, so either can be loaded by the same code.

    `iters` is carried so a resumed run keeps numbering gradient steps where it
    left off; checkpoints written before it existed load with iters treated as 0.
    """
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses,
                'iters': iters,
                }, path)
    # Companion copy without the Adam state, which is two buffers per parameter and so
    # two thirds of the file: ~3x smaller, which is what gets a checkpoint under
    # GitHub's 100 MiB cap. Inference reads model_state_dict only. The _noopt suffix is
    # load-bearing -- latest_checkpoint() matches model_epoch(N).pt with re.fullmatch,
    # so these can never be picked up as something to resume from. See strip_optimizer.py.
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'loss': losses, 'iters': iters}, path[:-len('.pt')] + '_noopt.pt')


def latest_checkpoint(output_dir):
    """Path of the highest-numbered model_epoch{N}.pt in output_dir, or None.

    Chosen by the epoch in the filename rather than by mtime, so re-copying files
    around does not change which one counts as most recent.
    """
    best, best_epoch = None, -1
    for name in os.listdir(output_dir):
        m = re.fullmatch(r"model_epoch(\d+)\.pt", name)
        if m and int(m.group(1)) > best_epoch:
            best, best_epoch = os.path.join(output_dir, name), int(m.group(1))
    return best


def resume_if_possible(model, optimizer, output_dir):
    """Continue from the newest checkpoint in output_dir, if there is one.

    Returns (epoch, iters, losses) to start from: the epoch AFTER the one the
    checkpoint recorded, since that epoch had already finished when it was written.
    Returns (0, 0, []) for a fresh run.
    """
    path = latest_checkpoint(output_dir)
    if path is None:
        print("No checkpoint found; starting from scratch.")
        return 0, 0, []
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    optimizer.load_state_dict(ck['optimizer_state_dict'])
    print(f"Resuming from {path}: finished epoch {ck['epoch']}, continuing at {ck['epoch']+1}")
    return ck['epoch'] + 1, ck.get('iters', 0), ck['loss']


f_log = open(f"{output_dir}/training.log", "a")

example_indices = [110, 198]

# Pick up where a previous run left off, if it left anything behind.
epoch, iters, losses = resume_if_possible(model, optimizer, output_dir)

print("Starting Training Loop...")
try:
    while (epoch < num_epochs) and (not break_flag):
        # For each batch in the dataloader
        err_total_loss = 0

        f_log.write(f"Epoch\t{epoch}.\n")

        for i, data in enumerate(train_loader, 0):
            err_total_loss += training_step(model, data, epoch, i, f_log, mode="train")

        f_log.write(f"Sum of losses over epochs (training): {err_total_loss}\n")
        f_log.write(f"Average of losses over epochs (training): {err_total_loss / train_size}\n")

        losses.append((epoch, -1, iters, {f"Average training loss": err_total_loss / train_size} ) )

        f_log.write(f"Epoch\t{epoch}. Evaluating validation loss\n")
        err_val_loss = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
            #while False:
                err_val_loss += training_step(model, data, epoch, i, f_log, mode="eval")
    
        f_log.write(f"Total validation loss: {err_val_loss}\n")
        f_log.write(f"Average validation loss: {err_val_loss / val_size}\n")

        losses.append((epoch, -1, iters, {f"Average validation loss": err_val_loss / val_size} ) )
        print(f"Loss info length: {len(losses)}")

        model.train()

        if (epoch+1) % 50 == 0:
            for example_idx in example_indices:
                save_example_illustration(model, dataset, example_idx, epoch, f_log)

            f_log.write(f"Saving checkpoint for epoch {epoch}\n")
            save_checkpoint(model, optimizer, epoch, losses, iters, f"{output_dir}/model_epoch{epoch}.pt")

        epoch += 1
        #SIGMA_RATIO = ((SIGMA_RATIO - 0.05) * 0.996) + 0.05
        #f_log.write(f"New SIGMA_RATIO: {SIGMA_RATIO}\n")

except KeyboardInterrupt:
    # Ctrl-C: keep the work done so far rather than discarding the epoch, then stop.
    # Exiting here deliberately skips the plotting and the model_final.pt save
    # below: a partial run should not overwrite the final checkpoint or the loss
    # plots of a completed one.
    path = f"{output_dir}/model_epoch{epoch}.pt"
    print(f"\nInterrupted during epoch {epoch}. Saving checkpoint...")
    f_log.write(f"Interrupted during epoch {epoch}. Saving checkpoint.\n")
    save_checkpoint(model, optimizer, epoch, losses, iters, path)
    print(f"Saved {path} -- rerun the same command to continue from here.")
    f_log.close()
    sys.exit(0)

f_log.close()


# --- cell 29
plot_losses(*losses_to_series(losses, batch_size), output_dir)


# --- cell 35
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses,
            }, f"{output_dir}/model_final.pt")