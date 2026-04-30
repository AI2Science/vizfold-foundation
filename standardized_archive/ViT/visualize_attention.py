import zarr
import matplotlib.pyplot as plt
import numpy as np

# Load archive

z = zarr.open("vit_trace.zarr", mode="r")
print(z.tree())

# Load image

img = z["inputs"]["processed_image"][:][0]   # (3,224,224)

img = np.transpose(img, (1,2,0))

img = (img * 0.5) + 0.5
img = np.clip(img, 0, 1)

plt.imshow(img)
plt.title("Input Image")
plt.axis("off")
plt.show()

# Show ViT patches

patch_size = 16

fig, axes = plt.subplots(14,14, figsize=(8,8))

for i in range(14):
    for j in range(14):

        patch = img[
            i*patch_size:(i+1)*patch_size,
            j*patch_size:(j+1)*patch_size
        ]

        axes[i,j].imshow(patch)
        axes[i,j].axis("off")

plt.suptitle("ViT Image Patches")
plt.show()

# Load attention

print(list(z["attention"].keys()))

layers = list(z["attention"].keys())

fig, axes = plt.subplots(3, 4, figsize=(12,9))

for i, layer in enumerate(layers):

    row = i // 4
    col = i % 4

    attn = z["attention"][layer][:][0]   # remove batch
    attn = attn.mean(axis=0)             # average heads

    cls_attn = attn[0,1:].reshape(14,14)
    cls_attn = cls_attn / cls_attn.max()

    heatmap = np.kron(cls_attn, np.ones((16,16)))

    axes[row,col].imshow(img)
    axes[row,col].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[row,col].set_title(layer)
    axes[row,col].axis("off")

plt.suptitle("ViT Attention Across Layers", fontsize=16)
plt.tight_layout()
plt.show()

# Predicted class

pred = z["outputs"].attrs["predicted_class"]
print("Predicted class index:", pred)

from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)

print("Predicted label:", model.config.id2label[pred])