"""Examine a trained vision transformer: load a checkpoint, run one dataset example
through it, and plot weights, predicted heatmaps, and per-layer activations.

Cells 45-64, split out of vit3.py so the examination can run without re-running
training. Reads tensors dumped during training from output_dir; writes every plot
to examine_dir.
"""

from vit_model import VisionTransformer
from custom_data_io import DataConfig, build_dataset

import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# --- cell 1
model_image_size = 128
output_dir = "model_output_020926_128x128_transformer"
examine_dir = f"{output_dir}/examine"
if not os.path.isdir(examine_dir):
    os.makedirs(examine_dir)
NUM_LANDMARKS = 5

config = DataConfig.load()
dataset = build_dataset(config)


# --- cell 45
model_dict = torch.load(f"{output_dir}/info_model_021026_872.pt")
model_loaded = VisionTransformer(model_image_size, 3, NUM_LANDMARKS, 4, 3)
model_loaded.load_state_dict(model_dict['model_state_dict'])


# --- cell 46
print(model_loaded)


# --- cell 47 (markdown)
# Goal: search for where outputs across channels collapse to one result.
# 
# Initial conv: 
# Patch conv: seemed that these weights changed, even after setting grad to false.
# MHA1:
# * query, key, value exhibit regular std (~0.1)
# * mlp weights and residual linear seem half as much as what they should be -- sqrt(2/256) -- correct bc of 0.4 factor added originally
#     * batchnorm2d weights range from 0.85 to 1.15, biases from -0.06 to 0.06
# * residual: std around 0.08
# * layer norm: 0.9 to 1.075 weight, -0.03 to 0.03 bias
# * combine: sounds good
# Found mistake in initialization scheme for Conv2D
# MHA2: no abn
# BatchNorm2D: no abn
# MHA3: no abn
# BatchNorm2D: no abns
# ConvTranspose2D: need to work out initialization scheme here.
# SameSizeConv: very last batchnorm has significant positive bias (0.2-0.7); otherwise, no abn


# --- cell 48
examined_weights = model_loaded.patch_conv.patch_filter.weight


# --- cell 49
print(examined_weights.shape)
plt.hist(examined_weights.flatten().detach().numpy(), bins=30)
plt.savefig(f"{examine_dir}/patch_filter_weights_hist.png")
plt.clf()


# --- cell 50
model_loaded.eval()


# --- cell 51
#DATA_PT = 115
DATA_PT = 179
example_point = dataset[DATA_PT]
input_example = example_point['image'].to(dtype=torch.float32)


# --- cell 52
image = torch.permute(example_point['image'], (1, 2, 0))
plt.imshow(image.numpy()*0.5 + 0.5)
plt.savefig(f"{examine_dir}/i{DATA_PT}_input.png")
plt.clf()


# --- cell 53
print(torch.min(input_example), torch.max(input_example))
print(input_example.shape)
SAVE_SWITCH=1
heatmap_pred = model_loaded( torch.stack((input_example,)) ).detach() # take unet outputted by first unet in the stacked hourglass
print(heatmap_pred.shape)
plt.imshow(heatmap_pred.numpy()[0][1])
plt.colorbar()
plt.savefig(f"{examine_dir}/i{DATA_PT}_heatmap1.png")
plt.clf()


# --- cell 54
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

# 4. Register hooks on desired layers
model_loaded.transformer_layers[0].register_forward_hook(get_activation('transformer_layers[0]')); dims['transformer_layers[0]'] = [1, 256,16]
model_loaded.transformer_layers[2].register_forward_hook(get_activation('transformer_layers[2]')); dims['transformer_layers[2]'] = [1, 256,16]
model_loaded.transformer_layers[4].register_forward_hook(get_activation('transformer_layers[4]')); dims['transformer_layers[4]'] = [1, 256,16]
model_loaded.patch_conv.register_forward_hook(get_activation('patch_conv')); dims['patch_conv'] = [1,192, 16]
model_loaded.resizechannels.register_forward_hook(get_activation('resizechannels')); dims['resizechannels'] = [1,256, 16]
model_loaded.transformer_layers[6].register_forward_hook(get_activation('up1')); dims['up1'] = [1,64,32]
model_loaded.transformer_layers[8].register_forward_hook(get_activation('up2')); dims['up2'] = [1,16,64]
model_loaded.transformer_layers[10].register_forward_hook(get_activation('up3')); dims['up3'] = [1,5,128]

# model_loaded.unet1.upsize1.register_forward_hook(get_activation('unet1.upsize1')); dims['unet1.upsize1'] = [1, 16, 128]
# model_loaded.unet1.upsize2.register_forward_hook(get_activation('unet1.upsize2')); dims['unet1.upsize2'] = [2, 16, 64]
# model_loaded.unet1.upsize3.register_forward_hook(get_activation('unet1.upsize3')); dims['unet1.upsize3'] = [3, 32, 32]
# model_loaded.unet1.upsize4.register_forward_hook(get_activation('unet1.upsize4')); dims['unet1.upsize4'] = [4, 64, 16]
# model_loaded.unet1.convs0a.register_forward_hook(get_activation('unet1.convs0a')); dims['unet1.convs0a'] = [0, 16, 128]
# model_loaded.unet1.convs1a.register_forward_hook(get_activation('unet1.convs1a')); dims['unet1.convs1a'] = [1, 16, 64]
# model_loaded.unet1.convs2a.register_forward_hook(get_activation('unet1.convs2a')); dims['unet1.convs2a'] = [2, 32, 32]
# model_loaded.unet1.convs3a.register_forward_hook(get_activation('unet1.convs3a')); dims['unet1.convs3a'] = [3, 64, 16]
# model_loaded.unet1.lowest_convs.register_forward_hook(get_activation('unet1.lowest_convs')); dims['unet1.lowest_convs'] = [-1, 128, 8]
# model_loaded.unet1.final_conv.register_forward_hook(get_activation('unet1.final_conv')); dims['unet1.final_conv'] = [-1, 13, 128]

# 5. Perform a forward pass
#sample_input = torch.randn(1, 3, 32, 32) # Batch size 1, 3 channels, 32x32 image
_ = model_loaded( torch.stack((input_example,)) ) # Run the model

# 6. Access the stored activations
print(f"Shape of mha1 output: {activations['transformer_layers[0]'].shape}")


# --- cell 55
def visualize_outputs(moniker, outputs):
    num_channels = outputs.shape[0]
    h = outputs.shape[1]
    w = outputs.shape[2]
    # Define the grid dimensions
    columns = 4
    rows = (num_channels+columns-1)//columns

    # find rank across output channels:
    output_mat = torch.flatten(outputs, start_dim=1).detach().numpy()
    print(output_mat.shape)
    s = np.linalg.svd(output_mat, compute_uv=False)
    # Select the top k singular values
    s_rank = [x for x in s if x > s[0]/100]
    print(f"\t{moniker}: Total Channels: {output_mat.shape[0]}, s_rank: {len(s_rank)}")

    # 2. Create figure and axes
    # fig is the entire figure, axs is a 2D array of axes (subplots)
    fig, axs = plt.subplots(rows, columns, figsize=(8, 3+rows))

    fig.suptitle(f"{moniker} features ({h}x{w}), srank={len(s_rank)}", fontsize=16, y=1)

    #images = [np.random.rand(10, 10) * i for i in range(1, 5)]
    vmin = torch.min(outputs).item()
    vmax = torch.max(outputs).item()
    # 2. Create the normalization and mappable
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis')

    titles = ["Channel "+str(i) for i in range(num_channels)]
    # 3. Iterate through axes and display images
    # Use .flatten() to turn the 2D array of axes into a 1D array for easy iteration
    for i, ax in enumerate(axs.flatten()):
        if i >= num_channels:
            break
        ax.imshow(outputs[i].numpy())
        ax.set_title(titles[i]) # Set a title for each subplot
        ax.axis('off') # Hide the axes ticks and labels for cleaner image presentation

    # Adjust layout for better spacing
    plt.tight_layout()
    fig.colorbar(sm, ax=axs, orientation='horizontal', label='Shared Scale')
    # Display the figure
    plt.savefig(f"{examine_dir}/{moniker}_{DATA_PT}.png")
    plt.close(fig)


# --- cell 56
for which_layer in dims:
    visualize_outputs(f"transformer_{which_layer}",  activations[which_layer][0])


# --- cell 57
visualize_outputs(f"transformer_mha1_raw_out", torch.load(f"{output_dir}/tsf_out_image.pt").detach()[0])
visualize_outputs(f"transformer_mha1_combined", torch.load(f"{output_dir}/post_res_norm.pt").detach()[0])
visualize_outputs(f"transformer_mha1_mlp_out", torch.load(f"{output_dir}/mlp_image.pt").detach()[0])
visualize_outputs(f"transformer_mha1_values", torch.load(f"{output_dir}/values.pt").detach()[0])


# --- cell 58
visualize_outputs(f"transformer_mha1_attentionmat", torch.load(f"{output_dir}/attention.pt").detach()[0])


# --- cell 59
visualize_outputs(f"transformer_mha1_cosinesimmat", torch.load(f"{output_dir}/cosine_sim.pt").detach()[0])


# --- cell 60
visualize_outputs(f"transformer_mha1_keys", torch.load(f"{output_dir}/keys.pt").detach()[0])


# --- cell 61
visualize_outputs(f"transformer_mha1_queries", torch.load(f"{output_dir}/queries.pt").detach()[0])


# --- cell 62
plt.hist(torch.load(f"{output_dir}/queryMatrix.pt").detach().numpy().flatten(), bins=30)
plt.savefig(f"{examine_dir}/queryMatrix_hist.png")
plt.clf()


# --- cell 63
plt.plot(range(256), np.linalg.svd(torch.load(f"{output_dir}/queryMatrix.pt").detach().numpy()[0], compute_uv=False))
plt.savefig(f"{examine_dir}/queryMatrix_svd.png")
plt.clf()


# --- cell 64
print(model_loaded.resizechannels)
print(model_loaded.resizechannels.weight.shape)
plt.hist(model_loaded.resizechannels.weight.detach().flatten(), bins=30)
plt.savefig(f"{examine_dir}/resizechannels_weights_hist.png")
plt.clf()