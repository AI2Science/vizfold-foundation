import zarr
import matplotlib.pyplot as plt

# root = zarr.open("mnist_trace.zarr", mode="r")

# image = root["inputs/images"][0][0]  # shape: (28, 28)

# plt.imshow(image, cmap="gray")
# plt.title("MNIST Image")
# plt.axis("off")
# plt.show()
# first shows the black and white MNIST image

###########################################

# activation = root["activations/conv1"][0]  # shape (16, 26, 26)

# Pick one filter (e.g., filter 0)
# feature_map = activation[0]

# plt.imshow(feature_map, cmap="viridis")
# plt.title("Conv1 Filter 0 Activation")
# plt.colorbar()
# plt.axis("off")
# plt.show()
#second shows the conv1 activation heatmap from filter 0

###########################################

import numpy as np

# activation = root["activations/conv1"][0]
# num_filters = activation.shape[0]

# fig, axes = plt.subplots(4, 4, figsize=(8, 8))

# for i, ax in enumerate(axes.flat):
#     ax.imshow(activation[i], cmap="viridis")
#     ax.axis("off")
#     ax.set_title(f"F{i}")

# plt.tight_layout()
# plt.show()
#third shows the 16 heatmaps, one for each filter

###########################################

# plt.imshow(image, cmap="gray")
# plt.imshow(feature_map, cmap="jet", alpha=0.5)
# plt.axis("off")
# plt.show()
#fourth shows the original digit and where the filter activtes on top of it

###########################################

# Normalize activation for better visualization
# feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())

# fig, ax = plt.subplots(figsize=(5,5))

# # Show original image
# ax.imshow(image, cmap="gray")

# # Overlay activation
# overlay = ax.imshow(feature_map, cmap="jet", alpha=0.5)

# ax.set_title("Original Digit + Conv1 Filter 0 Activation")
# ax.axis("off")

# # Add colorbar as key
# cbar = plt.colorbar(overlay, ax=ax)
# cbar.set_label("Activation Intensity")

# plt.show()

###########################################

root = zarr.open("mnist_trace.zarr", mode="r")

image = root["inputs/images"][0][0]

activation = root["activations/conv1"][0]

global_min = activation.min()
global_max = activation.max()

fig, axes = plt.subplots(4, 4, figsize=(8,8))

for i, ax in enumerate(axes.flat):
    fmap = activation[i]
    ax.imshow(image, cmap="gray")
    
    im = ax.imshow(
        fmap, 
        cmap="jet", 
        alpha=0.5,
        vmin=global_min, 
        vmax=global_max
    )
    ax.set_title(f"Filter {i}")
    ax.axis("off")

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Activation Intensity")

plt.show()
