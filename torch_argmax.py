import torch
import torch.nn.functional as F

# Assume 'logits' is your tensor of raw logits from a model output
# For example, a batch of 3 samples with 4 possible classes
logits = torch.tensor([
    [[[1.2, 0.5, 2.1, 0.8], [1.2, 0.5, 2.1, 0.8]],  # Sample 1
    [[0.1, 3.5, 0.9, 1.0], [1.2, 0.5, 2.1, 0.8]],  # Sample 2
    [[2.0, 1.8, 0.3, 1.5], [1.2, 0.5, 2.1, 0.8]]
    ]   # Sample 3
])

print("Original logits")
print(logits)

probs = F.softmax(logits / 0.001, dim=-3)

print(probs.shape)

# Step 1: Get the predicted class index for each sample
# torch.argmax returns the index of the maximum value along the specified dimension.
# dim=-1 indicates the last dimension (which represents the classes).
predicted_classes = torch.argmax(torch.permute(probs, (0, 2, 3, 1)), dim=-1)

# Step 2: Convert the class indices to one-hot vectors
# num_classes should be the total number of possible classes.
num_classes = probs.shape[-3]

probs_tr = torch.permute(probs, (0, 2, 3, 1))

print("Transposed probs")
print(probs_tr)
categories = torch.reshape(torch.multinomial(torch.reshape(probs_tr, (-1, num_classes)), 1, replacement=True), probs_tr.shape[:-1])

print("Categories")
print(categories)

one_hot_vectors = torch.permute(F.one_hot(categories, num_classes=num_classes), (0, 3, 1, 2))
#one_hot_vectors = F.one_hot(categories, num_classes=num_classes)

print("Original Logits:")
print(logits)
print("\nPredicted Class Indices:")
print(predicted_classes)
print("\nOne-Hot Encoded Vectors:")
print(one_hot_vectors)

#print(torch.mean(one_hot_vectors.to(dtype=torch.float32), dim=(0, 2, 3)))