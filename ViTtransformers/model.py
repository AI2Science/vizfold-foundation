"""
Vision Transformer (ViT) implemented from scratch for MNIST.

Architecture follows "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020),
adapted for small 28x28 grayscale images using 4x4 patches.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension."""

    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=64):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        # Conv2d with kernel=patch_size, stride=patch_size extracts non-overlapping patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # average_attn_weights=False → shape (B, num_heads, seq_len, seq_len)
        attn_out, attn_weights = self.attn(x, x, x, average_attn_weights=False)
        return self.norm(x + attn_out), attn_weights


class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x, attn_weights = self.attn(x)
        x = self.ff(x)
        return x, attn_weights


class ViT(nn.Module):
    """
    Vision Transformer for MNIST classification.

    Args:
        img_size:    Input image size (default 28 for MNIST).
        patch_size:  Patch size (default 4, giving 7x7=49 patches).
        in_channels: Number of input channels (1 for grayscale).
        num_classes: Number of output classes (10 for MNIST).
        embed_dim:   Token embedding dimension.
        depth:       Number of transformer blocks.
        num_heads:   Number of attention heads.
        mlp_ratio:   MLP hidden dimension multiplier.
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        img_size=28,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        depth=6,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        attn_weights_all = []
        for block in self.blocks:
            x, attn_w = block(x)
            attn_weights_all.append(attn_w)

        x = self.norm(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)

        return logits, attn_weights_all
