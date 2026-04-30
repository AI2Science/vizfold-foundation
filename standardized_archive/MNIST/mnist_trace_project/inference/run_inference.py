import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import SimpleCNN
from utils.hooks import register_hooks
from archive.schema import create_archive
from archive.writer import write_batch

def run(output_path):

    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64)

    model = SimpleCNN()
    model.eval()

    activations = register_hooks(model)

    root = create_archive(output_path, len(dataset))

    start_idx = 0

    with torch.no_grad():
        for images, labels in dataloader:
            logits = model(images)
            preds = logits.argmax(dim=1)

            start_idx = write_batch(
                root,
                start_idx,
                images.numpy(),
                logits.numpy(),
                preds.numpy(),
                activations
            )