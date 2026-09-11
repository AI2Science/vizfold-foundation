"""
Visualize ViT outputs saved by train.py.

Produces three figures in the output directory:
  predictions.png       — sample images with true vs predicted labels
  attention_rollout.png — attention rollout heatmap overlaid on each image
  attention_heads.png   — per-head attention from the last transformer block

Usage:
    python visualize.py
    python visualize.py --input outputs/sample_attention.pt --out-dir outputs/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# MNIST normalization constants (must match train.py)
MEAN = 0.1307
STD  = 0.3081


def denormalize(img_tensor):
    """(1, H, W) normalized tensor -> (H, W) numpy array in [0, 1]."""
    img = img_tensor.squeeze().numpy()
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


def attention_rollout(attn_weights_list):
    """
    Compute attention rollout for a single sample.

    Args:
        attn_weights_list: list of tensors, each (num_heads, seq_len, seq_len)

    Returns:
        rollout: (seq_len, seq_len) numpy array
    """
    rollout = None
    for attn in attn_weights_list:
        # Average over heads
        A = attn.mean(dim=0).numpy()               # (seq_len, seq_len)
        # Add residual identity
        A_hat = 0.5 * A + 0.5 * np.eye(A.shape[0])
        # Normalize rows
        A_hat /= A_hat.sum(axis=-1, keepdims=True)
        if rollout is None:
            rollout = A_hat
        else:
            rollout = A_hat @ rollout
    return rollout


def cls_to_patch_map(attn_row, patch_grid=7, img_size=28):
    """
    Convert CLS-to-patches attention row to an upsampled heatmap.

    Args:
        attn_row:   1-D array of length num_patches (CLS token excluded)
        patch_grid: sqrt of num_patches (7 for our 28x28 / patch4 setup)
        img_size:   final output size

    Returns:
        heatmap: (img_size, img_size) numpy array in [0, 1]
    """
    heatmap = attn_row.reshape(patch_grid, patch_grid)
    heatmap = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float()
    heatmap = F.interpolate(heatmap, size=(img_size, img_size), mode="bilinear", align_corners=False)
    heatmap = heatmap.squeeze().numpy()
    # Normalize to [0, 1]
    lo, hi = heatmap.min(), heatmap.max()
    if hi > lo:
        heatmap = (heatmap - lo) / (hi - lo)
    return heatmap


# ---------------------------------------------------------------------------
# Figure 1: Predictions
# ---------------------------------------------------------------------------

def plot_predictions(images, labels, preds, out_path):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))
    for i, ax in enumerate(axes):
        img = denormalize(images[i])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        correct = labels[i].item() == preds[i].item()
        color = "green" if correct else "red"
        ax.set_title(f"T:{labels[i].item()}\nP:{preds[i].item()}", color=color, fontsize=9)
    fig.suptitle("Sample Predictions  (green=correct, red=wrong)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Attention Rollout
# ---------------------------------------------------------------------------

def plot_attention_rollout(images, labels, preds, attn_weights, out_path):
    n = len(images)
    num_layers = len(attn_weights)    # 6
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))

    for i, ax in enumerate(axes):
        img = denormalize(images[i])

        # attn_weights[layer] shape: (B, heads, seq, seq) — pick sample i
        sample_attn = [attn_weights[l][i] for l in range(num_layers)]
        rollout = attention_rollout(sample_attn)   # (seq_len, seq_len)

        # CLS row, drop CLS column → patch attention
        cls_attn = rollout[0, 1:]                  # (49,)
        heatmap = cls_to_patch_map(cls_attn)       # (28, 28)

        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.imshow(heatmap, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        ax.axis("off")
        correct = labels[i].item() == preds[i].item()
        color = "green" if correct else "red"
        ax.set_title(f"T:{labels[i].item()}  P:{preds[i].item()}", color=color, fontsize=8)

    fig.suptitle("Attention Rollout — what the CLS token attends to (all layers)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Per-Head Attention (last layer)
# ---------------------------------------------------------------------------

def plot_per_head_attention(images, labels, preds, last_layer_attn, out_path):
    """
    last_layer_attn: (B, num_heads, seq_len, seq_len)
    """
    n = images.shape[0]
    num_heads = last_layer_attn.shape[1]

    fig, axes = plt.subplots(n, num_heads, figsize=(2 * num_heads, 2 * n))

    for i in range(n):
        img = denormalize(images[i])
        for h in range(num_heads):
            ax = axes[i, h]
            head_attn = last_layer_attn[i, h].numpy()  # (seq_len, seq_len)
            cls_attn = head_attn[0, 1:]                # CLS row, drop CLS col → 49 patches
            heatmap = cls_to_patch_map(cls_attn)

            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.imshow(heatmap, cmap="hot", alpha=0.5, vmin=0, vmax=1)
            ax.axis("off")

            if i == 0:
                ax.set_title(f"Head {h}", fontsize=8)
            if h == 0:
                correct = labels[i].item() == preds[i].item()
                color = "green" if correct else "red"
                ax.set_ylabel(f"T:{labels[i].item()} P:{preds[i].item()}", color=color, fontsize=7)

    fig.suptitle("Last-Layer Per-Head Attention", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",   default="./outputs/sample_attention.pt")
    p.add_argument("--out-dir", default="./outputs")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Run train.py first.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    data = torch.load(args.input, map_location="cpu", weights_only=False)
    images       = data["images"]        # (8, 1, 28, 28)
    labels       = data["labels"]        # (8,)
    preds        = data["preds"]         # (8,)
    attn_weights = data["attn_weights"]  # list of 6 x (8, heads, seq, seq)

    plot_predictions(
        images, labels, preds,
        os.path.join(args.out_dir, "predictions.png"),
    )

    plot_attention_rollout(
        images, labels, preds, attn_weights,
        os.path.join(args.out_dir, "attention_rollout.png"),
    )

    plot_per_head_attention(
        images, labels, preds,
        attn_weights[-1],   # last transformer block
        os.path.join(args.out_dir, "attention_heads.png"),
    )


if __name__ == "__main__":
    main()