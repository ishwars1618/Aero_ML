import torch
import torch.nn.functional as F

# Input tensor (batch_size, in_channels, height, width)
input_tensor = torch.randn(1, 1, 5, 5)

# Define a fixed 3x3 kernel (e.g., a simple edge detection kernel)
# Shape: (out_channels, in_channels/groups, kernel_height, kernel_width)
fixed_kernel = torch.tensor([
    [-1., -1., -1.],
    [-1.,  8., -1.],
    [-1., -1., -1.]
]).view(1, 1, 3, 3) # Reshape for conv2d
# Ensure the kernel is not learnable
fixed_kernel.requires_grad_(False)

# Perform the convolution
output_tensor = F.conv2d(input_tensor, fixed_kernel, padding='same')

print("Input tensor shape:", input_tensor.shape)
print("Fixed kernel shape:", fixed_kernel.shape)
print("Output tensor shape:", output_tensor.shape)
print("Input tensor:\n", input_tensor)
print("Output tensor:\n", output_tensor)