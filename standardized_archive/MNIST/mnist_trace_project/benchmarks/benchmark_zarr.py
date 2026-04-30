import time
import zarr

def benchmark(path):
    root = zarr.open(path, mode="r")

    start = time.time()
    _ = root["activations/conv1"][500]
    print("Random read time:", time.time() - start)