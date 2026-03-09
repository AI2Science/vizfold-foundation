"""
Train ViT on MNIST and save attention weights + predictions.

Usage:
    python train.py                    # default settings
    python train.py --epochs 20 --device cuda
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ViT


def get_args():
    p = argparse.ArgumentParser(description="ViT MNIST trainer")
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--device",     type=str,   default="auto",
                   help="'auto', 'cpu', or 'cuda'")
    p.add_argument("--data-dir",   type=str,   default="./data")
    p.add_argument("--save-dir",   type=str,   default="./outputs")
    # ViT hyperparams
    p.add_argument("--patch-size", type=int,   default=4)
    p.add_argument("--embed-dim",  type=int,   default=64)
    p.add_argument("--depth",      type=int,   default=6)
    p.add_argument("--num-heads",  type=int,   default=4)
    p.add_argument("--dropout",    type=float, default=0.1)
    return p.parse_args()


def build_dataloaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _ = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def save_sample_attention(model, loader, device, save_dir, num_samples=8):
    """Save attention weights and sample images for later visualization."""
    model.eval()
    imgs, labels = next(iter(loader))
    imgs, labels = imgs[:num_samples].to(device), labels[:num_samples]

    logits, attn_weights_all = model(imgs)
    preds = logits.argmax(1).cpu()

    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "images":       imgs.cpu(),
        "labels":       labels,
        "preds":        preds,
        # list of tensors: (depth, B, num_heads, seq_len, seq_len)
        "attn_weights": [w.cpu() for w in attn_weights_all],
    }, os.path.join(save_dir, "sample_attention.pt"))
    print(f"Saved attention sample to {save_dir}/sample_attention.pt")


def main():
    args = get_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size)

    model = ViT(
        img_size=28,
        patch_size=args.patch_size,
        in_channels=1,
        num_classes=10,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ViT parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Test  loss: {test_loss:.4f}, acc: {test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))

    print(f"\nBest test accuracy: {best_acc:.4f}")
    save_sample_attention(model, test_loader, device, args.save_dir)


if __name__ == "__main__":
    main()
