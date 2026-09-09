"""Examine a trained U-Net: load a checkpoint, run one dataset example through it,
and plot the per-channel activations of each named layer.

Cells 36-43, split out of train_model.py so the examination can run
without re-running training.
"""

from unet_model import StackedHourglass
from custom_data_io import DataConfig, build_dataset

import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# --- cell 1
output_dir = "model_output_090526_128x128_unet"
examine_dir = f"{output_dir}/examine"
if not os.path.isdir(examine_dir):
    os.makedirs(examine_dir)
NUM_LANDMARKS = 5

config = DataConfig.load()
dataset = build_dataset(config)


# --- cell 36
model_dict = torch.load(f"{output_dir}/info_model_060626_1000.pt")
model_loaded = StackedHourglass(3, NUM_LANDMARKS)
model_loaded.load_state_dict(model_dict['model_state_dict'])

# --- cell 38
model_loaded.eval()


# --- cell 39
DATA_PT = 179
example_point = dataset[DATA_PT]
input_example = example_point['image'].to(dtype=torch.float32)


# --- cell 40
image = torch.permute(example_point['image'], (1, 2, 0))
plt.imshow(image.numpy()*0.5 + 0.5)
plt.savefig(f"{examine_dir}/i{DATA_PT}_input.png")
plt.clf()


# --- cell 41
print(torch.min(input_example), torch.max(input_example))
print(input_example.shape)
heatmap_pred = model_loaded( torch.stack((input_example,)) )[0].detach()[0] # take unet outputted by first unet in the stacked hourglass
print(heatmap_pred.shape)
plt.imshow(heatmap_pred.numpy()[1])
plt.colorbar()
plt.savefig(f"{examine_dir}/i{DATA_PT}_heatmap1.png")
plt.clf()


# --- cell 42
# provided by Gemini

# 2. Create a dictionary to store activations
activations = {}

# 3. Define the hook function
def get_activation(name):
    def hook(model, input, output):
        # .detach() removes the tensor from the computation graph
        activations[name] = output.detach()
    return hook

# 4. Register hooks on desired layers
dims = {}
model_loaded.unet1.upsize1.register_forward_hook(get_activation('unet1.upsize1')); dims['unet1.upsize1'] = [1, 16, 128]
model_loaded.unet1.upsize2.register_forward_hook(get_activation('unet1.upsize2')); dims['unet1.upsize2'] = [2, 16, 64]
model_loaded.unet1.upsize3.register_forward_hook(get_activation('unet1.upsize3')); dims['unet1.upsize3'] = [3, 32, 32]
model_loaded.unet1.upsize4.register_forward_hook(get_activation('unet1.upsize4')); dims['unet1.upsize4'] = [4, 64, 16]
model_loaded.unet1.convs0a.register_forward_hook(get_activation('unet1.convs0a')); dims['unet1.convs0a'] = [0, 16, 128]
model_loaded.unet1.convs1a.register_forward_hook(get_activation('unet1.convs1a')); dims['unet1.convs1a'] = [1, 16, 64]
model_loaded.unet1.convs2a.register_forward_hook(get_activation('unet1.convs2a')); dims['unet1.convs2a'] = [2, 32, 32]
model_loaded.unet1.convs3a.register_forward_hook(get_activation('unet1.convs3a')); dims['unet1.convs3a'] = [3, 64, 16]
model_loaded.unet1.lowest_convs.register_forward_hook(get_activation('unet1.lowest_convs')); dims['unet1.lowest_convs'] = [-1, 128, 8]
model_loaded.unet1.final_conv.register_forward_hook(get_activation('unet1.final_conv')); dims['unet1.final_conv'] = [-1, 13, 128]

# 5. Perform a forward pass
#sample_input = torch.randn(1, 3, 32, 32) # Batch size 1, 3 channels, 32x32 image
_ = model_loaded( torch.stack((input_example,)) ) # Run the model

# 6. Access the stored activations
print(f"Shape of conv1 output: {activations['unet1.upsize1'].shape}")


# --- cell 43
for which_layer in dims:
    num_channels = dims[which_layer][1]
    im_dim = dims[which_layer][2]
    # Define the grid dimensions
    columns = 4
    rows = (num_channels+columns-1)//columns

    # 2. Create figure and axes
    # fig is the entire figure, axs is a 2D array of axes (subplots)
    fig, axs = plt.subplots(rows, columns, figsize=(8, 3+rows))

    fig.suptitle(f"{which_layer} features ({im_dim}x{im_dim})", fontsize=16, y=1)

    images = [np.random.rand(10, 10) * i for i in range(1, 5)]
    vmin = torch.min(activations[which_layer]).item()
    vmax = torch.max(activations[which_layer]).item()
    # 2. Create the normalization and mappable
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis')

    titles = ["Channel "+str(i) for i in range(num_channels)]
    # 3. Iterate through axes and display images
    # Use .flatten() to turn the 2D array of axes into a 1D array for easy iteration
    for i, ax in enumerate(axs.flatten()):
        if i >= num_channels:
            break
        ax.imshow(activations[which_layer][0][i].numpy())
        ax.set_title(titles[i]) # Set a title for each subplot
        ax.axis('off') # Hide the axes ticks and labels for cleaner image presentation

    # Adjust layout for better spacing
    plt.tight_layout()
    fig.colorbar(sm, ax=axs, orientation='horizontal', label='Shared Scale')
    # Display the figure
    plt.savefig(f"{examine_dir}/{which_layer}_{DATA_PT}.png")
    plt.close(fig)