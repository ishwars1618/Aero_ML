"""Loss curves for a training run.

Split out of train_model.py so the same plots can be produced after the fact from a
saved checkpoint (see plot_checkpoint.py) without re-running or importing training.

`losses` is the list the training loop accumulates and save_checkpoint stores under
the 'loss' key: (epoch, batch, iters, {label: value}) tuples, where batch == -1 marks
the per-epoch averages and anything else is a single gradient step.
"""

import math

import matplotlib.pyplot as plt


def losses_to_series(losses, batch_size):
    """Split the raw losses list into the five series the plots need.

    Was cell 29 of the notebook, inline in train_model.py.
    """
    validation_losses = [x[3]["Average validation loss"]*batch_size for x in losses if ((x[1] == -1) and ("Average validation loss" in x[3]))]
    training_losses = [x[3]["Average training loss"]*batch_size for x in losses if ((x[1] == -1) and ("Average training loss" in x[3]))]
    epoch_nums = [x[0] for x in losses if ((x[1] == -1) and ("Average training loss" in x[3]))]

    batch_losses = [x[3]["Stage 0 error"] for x in losses if ((x[1] != -1) and ("Stage 0 error" in x[3]))]
    iter_nums = [x[2] for x in losses if ((x[1] != -1) and ("Stage 0 error" in x[3]))]
    return epoch_nums, validation_losses, training_losses, iter_nums, batch_losses


def plot_losses(epoch_nums, validation_losses, training_losses, iter_nums, batch_losses, output_dir):
    """Per-epoch training/validation loss and per-step batch loss, linear and log.

    Saved as PNGs under output_dir rather than shown: plt.show() opens a window and
    blocks until it is closed, which stalls an unattended run.

    epoch_nums  -- x values for validation_losses and training_losses
    iter_nums   -- x values for batch_losses
    output_dir  -- directory the four PNGs are written to
    """
    # --- cell 30
    print(validation_losses[-5:])
    print(training_losses[-5:])
    print(epoch_nums[-5:])
    print(batch_losses[-5:])
    print(iter_nums[-5:])


    # --- cell 31
    plt.title("Training loss vs Validation loss (softmax->MSE)")
    plt.plot(epoch_nums, validation_losses, label="Validation loss")
    plt.plot(epoch_nums, training_losses, label="Training loss")
    plt.ylim(bottom=0)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/loss_vs_epoch.png")
    plt.clf()


    # --- cell 32
    plt.title("Log training loss vs log validation loss (softmax->MSE)")
    plt.plot(epoch_nums, [math.log(x, 10) for x in validation_losses], label="Log validation loss")
    plt.plot(epoch_nums, [math.log(x, 10) for x in training_losses], label="Log training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/log_loss_vs_epoch.png")
    plt.clf()


    # --- cell 33
    plt.title("Loss per gradient step (softmax->MSE)")
    plt.plot(iter_nums, batch_losses, label="Loss / iteration")
    plt.xlabel("Gradient steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/loss_per_step.png")
    plt.clf()


    # --- cell 34
    plt.title("Log Loss per gradient step (softmax->MSE)")
    plt.plot(iter_nums, [math.log(x, 10) for x in batch_losses], label="Log loss / iteration")
    plt.xlabel("Gradient steps")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/log_loss_per_step.png")
    plt.clf()
