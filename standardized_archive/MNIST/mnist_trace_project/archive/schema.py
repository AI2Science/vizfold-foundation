import zarr
from numcodecs import Blosc

def create_archive(path, dataset_size):
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)

    root = zarr.open(path, mode="w")

    root.create_dataset(
        "inputs/images",
        shape=(dataset_size, 1, 28, 28),
        chunks=(64, 1, 28, 28),
        dtype="float32",
        compressor=compressor,
    )

    root.create_dataset(
        "outputs/logits",
        shape=(dataset_size, 10),
        chunks=(64, 10),
        dtype="float32",
        compressor=compressor,
    )

    root.create_dataset(
        "outputs/predictions",
        shape=(dataset_size,),
        chunks=(64,),
        dtype="int64",
        compressor=compressor,
    )

    # Activations (example shapes — update after first forward pass if needed)
    root.create_dataset(
        "activations/conv1",
        shape=(dataset_size, 16, 26, 26),
        chunks=(64, 16, 26, 26),
        dtype="float32",
        compressor=compressor,
    )

    root.attrs["dataset"] = "MNIST"
    root.attrs["archive_version"] = "v1"

    return root