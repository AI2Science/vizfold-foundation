import argparse
from pathlib import Path

import torch


def summarize_tensor(name: str, value: torch.Tensor):
    value_f = value.float()
    print(
        f"{name}: "
        f"shape={tuple(value.shape)}, "
        f"dtype={value.dtype}, "
        f"mean={float(value_f.mean()):.5f}, "
        f"std={float(value_f.std()):.5f}, "
        f"min={float(value_f.min()):.5f}, "
        f"max={float(value_f.max()):.5f}"
    )


def compare_tensors(payload, key_a: str, key_b: str):
    if key_a not in payload or key_b not in payload:
        print(f"Skipping comparison {key_a} vs {key_b} (missing key)")
        return

    a = payload[key_a]
    b = payload[key_b]

    print(f"Comparing {key_a} vs {key_b}")
    print(f"same_shape={a.shape == b.shape}")
    if a.shape == b.shape:
        print(f"allclose={torch.allclose(a, b)}")
    print()


def main(path: str):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    payload = torch.load(path, map_location="cpu")

    print(f"Loaded: {path}")
    print(f"Number of keys: {len(payload)}")
    print()

    for key in sorted(payload.keys()):
        value = payload[key]
        if isinstance(value, torch.Tensor):
            summarize_tensor(key, value)
        else:
            print(f"{key}: type={type(value)}")

    print()
    print("Layer comparison check:")
    compare_tensors(payload, "layer_00.msa", "layer_47.msa")
    compare_tensors(payload, "layer_00.pair", "layer_47.pair")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to saved evoformer reps .pt file")
    args = parser.parse_args()
    main(args.path)
