import os
import time
import json
import shutil
import random
import numpy as np
import concurrent.futures
import zarr
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from numcodecs import Blosc
from tqdm import tqdm


# Global styling (clean paper look)
plt.rcParams.update({
    "figure.figsize": (6, 4),
    "axes.grid": True,
    "font.size": 11
})

# -----------------------------
# CONFIGURATION
# -----------------------------
CONFIG = {
    "N_res": 256,
    "num_layers": 12,
    "num_heads": 8,
    "hidden_dim": 384,
    "pair_dim": 128,
    "dtype": "float32",
    "runs": 3,
    "output_dir": "benchmark_data"
}

np.random.seed(42)
random.seed(42)

# -----------------------------
# DATA GENERATION
# -----------------------------
def generate_data(cfg):
    N = cfg["N_res"]
    L = cfg["num_layers"]
    H = cfg["num_heads"]

    data = {
        "metadata": {
            "num_layers": L,
            "num_heads": H,
            "N_res": N
        },
        "activations": [],
        "attention": [],
        "outputs": {}
    }

    for l in range(L):
        data["activations"].append({
            "single": np.random.randn(N, cfg["hidden_dim"]).astype(cfg["dtype"]),
            "pair": np.random.randn(N, N, cfg["pair_dim"]).astype(cfg["dtype"])
        })

        data["attention"].append(
            np.random.randn(H, N, N).astype(cfg["dtype"])
        )

    data["outputs"]["coordinates"] = np.random.randn(N, 3).astype(cfg["dtype"])
    data["outputs"]["confidence"] = np.random.randn(N).astype(cfg["dtype"])

    return data

import concurrent.futures
import random
import time
import zarr
import h5py

# -----------------------------
# PARALLEL ACCESS TEST
# -----------------------------
def zarr_worker(root, cfg):
    l = random.randint(0, cfg["num_layers"] - 1)
    h = random.randint(0, cfg["num_heads"] - 1)
    i = random.randint(0, cfg["N_res"] - 32)
    return root[f"attention/layer_{l}"][h, i:i+32, i:i+32]


def hdf5_worker(path, cfg):
    # each worker opens its own handle (realistic multi-process behavior)
    with h5py.File(path, "r") as f:
        l = random.randint(0, cfg["num_layers"] - 1)
        h = random.randint(0, cfg["num_heads"] - 1)
        i = random.randint(0, cfg["N_res"] - 32)
        return f[f"attention/layer_{l}"][h, i:i+32, i:i+32]


def benchmark_parallel_access(cfg, num_workers=8, num_tasks=100):
    results = {}

    zarr_path = os.path.join(cfg["output_dir"], "data.zarr")
    h5_path = os.path.join(cfg["output_dir"], "data.h5")

    root = zarr.open(zarr_path, mode="r")

    # -----------------------------
    # ZARR PARALLEL
    # -----------------------------
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(zarr_worker, root, cfg) for _ in range(num_tasks)]
        _ = [f.result() for f in futures]
    results["zarr_parallel"] = time.time() - start

    # -----------------------------
    # HDF5 PARALLEL
    # -----------------------------
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(hdf5_worker, h5_path, cfg) for _ in range(num_tasks)]
        _ = [f.result() for f in futures]
    results["hdf5_parallel"] = time.time() - start

    return results

# -----------------------------
# NPZ
# -----------------------------
def write_npz(data, path):
    flat = {}
    for i, layer in enumerate(data["activations"]):
        flat[f"act_{i}_single"] = layer["single"]
        flat[f"act_{i}_pair"] = layer["pair"]
        flat[f"attn_{i}"] = data["attention"][i]

    flat["coords"] = data["outputs"]["coordinates"]
    flat["conf"] = data["outputs"]["confidence"]

    np.savez_compressed(path, **flat)


def read_npz_full(path):
    return np.load(path)


# -----------------------------
# HDF5
# -----------------------------
def write_hdf5(data, path):
    with h5py.File(path, "w") as f:
        for i, layer in enumerate(data["activations"]):
            grp = f.create_group(f"activations/layer_{i}")
            grp.create_dataset("single", data=layer["single"])
            grp.create_dataset("pair", data=layer["pair"])
            f.create_dataset(f"attention/layer_{i}", data=data["attention"][i])

        f.create_dataset("outputs/coords", data=data["outputs"]["coordinates"])
        f.create_dataset("outputs/conf", data=data["outputs"]["confidence"])


def read_hdf5_full(path):
    with h5py.File(path, "r") as f:
        _ = f["outputs/coords"][:]


# -----------------------------
# ZARR
# -----------------------------
def write_zarr(data, path, chunk_size=32):
    root = zarr.open(path, mode="w")
    compressor = Blosc(cname="zstd", clevel=3)

    for i, layer in enumerate(data["activations"]):
        grp = root.create_group(f"activations/layer_{i}")

        # single representation
        grp.create_dataset(
            "single",
            data=layer["single"],
            chunks=(chunk_size, layer["single"].shape[1]),
            compressor=compressor
        )

        # pair representation
        grp.create_dataset(
            "pair",
            data=layer["pair"],
            chunks=(chunk_size, chunk_size, layer["pair"].shape[2]),
            compressor=compressor
        )

        # attention
        root.create_dataset(
            f"attention/layer_{i}",
            data=data["attention"][i],
            chunks=(1, chunk_size, chunk_size),
            compressor=compressor
        )

    # outputs
    root.create_dataset(
        "outputs/coords",
        data=data["outputs"]["coordinates"],
        compressor=compressor
    )
    root.create_dataset(
        "outputs/conf",
        data=data["outputs"]["confidence"],
        compressor=compressor
    )

def read_zarr_full(path):
    root = zarr.open(path, mode="r")
    _ = root["outputs/coords"][:]


# -----------------------------
# BENCHMARK HELPERS
# -----------------------------
def time_fn(fn, *args):
    start = time.time()
    fn(*args)
    return time.time() - start

def get_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    else:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total

def benchmark_write(data, cfg):
    results = {}

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # NPZ
    npz_path = os.path.join(cfg["output_dir"], "data.npz")
    results["npz_write"] = time_fn(write_npz, data, npz_path)
    results["npz_size"] = get_size(npz_path)

    # HDF5
    h5_path = os.path.join(cfg["output_dir"], "data.h5")
    results["hdf5_write"] = time_fn(write_hdf5, data, h5_path)
    results["hdf5_size"] = get_size(h5_path)

    # ZARR
    zarr_path = os.path.join(cfg["output_dir"], "data.zarr")
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    results["zarr_write"] = time_fn(write_zarr, data, zarr_path, 32)
    results["zarr_size"] = get_size(zarr_path)

    return results


def benchmark_reads(cfg):
    results = {}

    npz_path = os.path.join(cfg["output_dir"], "data.npz")
    h5_path = os.path.join(cfg["output_dir"], "data.h5")
    zarr_path = os.path.join(cfg["output_dir"], "data.zarr")

    # FULL READ
    results["npz_full_read"] = time_fn(read_npz_full, npz_path)
    results["hdf5_full_read"] = time_fn(read_hdf5_full, h5_path)
    results["zarr_full_read"] = time_fn(read_zarr_full, zarr_path)

    # PARTIAL READ (layer)
    layer = random.randint(0, CONFIG["num_layers"] - 1)

    # NPZ (loads everything)
    start = time.time()
    d = np.load(npz_path)
    _ = d[f"attn_{layer}"]
    results["npz_layer_read"] = time.time() - start

    # HDF5
    with h5py.File(h5_path, "r") as f:
        start = time.time()
        _ = f[f"attention/layer_{layer}"][:]
        results["hdf5_layer_read"] = time.time() - start

    # ZARR
    root = zarr.open(zarr_path, mode="r")
    start = time.time()
    _ = root[f"attention/layer_{layer}"][:]
    results["zarr_layer_read"] = time.time() - start

    # HEAD READ
    head = random.randint(0, CONFIG["num_heads"] - 1)

    with h5py.File(h5_path, "r") as f:
        start = time.time()
        _ = f[f"attention/layer_{layer}"][head]
        results["hdf5_head_read"] = time.time() - start

    start = time.time()
    _ = root[f"attention/layer_{layer}"][head]
    results["zarr_head_read"] = time.time() - start

    # RANDOM ACCESS LOOP
    def random_access_zarr():
        for _ in range(500):
            l = random.randint(0, CONFIG["num_layers"] - 1)
            h = random.randint(0, CONFIG["num_heads"] - 1)
            i = random.randint(0, CONFIG["N_res"] - 32)
            _ = root[f"attention/layer_{l}"][h, i:i+32, i:i+32]

    def random_access_hdf5():
        with h5py.File(h5_path, "r") as f:
            for _ in range(500):
                l = random.randint(0, CONFIG["num_layers"] - 1)
                h = random.randint(0, CONFIG["num_heads"] - 1)
                i = random.randint(0, CONFIG["N_res"] - 32)
                _ = f[f"attention/layer_{l}"][h, i:i+32, i:i+32]

    results["zarr_random"] = time_fn(random_access_zarr)
    results["hdf5_random"] = time_fn(random_access_hdf5)

    return results


def viz_simulation_zarr(root, cfg):
    L = cfg["num_layers"]
    H = cfg["num_heads"]
    N = cfg["N_res"]

    start = time.time()

    for l in range(L):
        for h in range(H):
            i = random.randint(0, N - 32)
            _ = root[f"attention/layer_{l}"][h, i:i+32, i:i+32]
    
    return time.time() - start

def viz_simulation_hdf5(path, cfg):
    L = cfg["num_layers"]
    H = cfg["num_heads"]
    N = cfg["N_res"]

    start = time.time()

    with h5py.File(path, "r") as f:
        for l in range(L):
            for h in range(H):
                i = random.randint(0, N - 32)
                _ = f[f"attention/layer_{l}"][h, i:i+32, i:i+32]

    return time.time() - start

# -----------------------------
# MAIN
# -----------------------------
def run_benchmark():
    base_cfg = CONFIG

    N_values = [256, 512, 1024, 2048]

    all_results = []

    for N in N_values:
        print(f"\n=== Running for N_res = {N} ===")

        cfg = base_cfg.copy()
        cfg["N_res"] = N

        data = generate_data(cfg)
        
        write_res = benchmark_write(data, cfg)

        npz_path = os.path.join(cfg["output_dir"], "data.npz")
        h5_path = os.path.join(cfg["output_dir"], "data.h5")
        zarr_path = os.path.join(cfg["output_dir"], "data.zarr")

        root = zarr.open(zarr_path, mode="r")
        read_res = benchmark_reads(cfg)

        read_res["viz_hdf5"] = viz_simulation_hdf5(h5_path, cfg)
        read_res["viz_zarr"] = viz_simulation_zarr(root, cfg)

        parallel_res = benchmark_parallel_access(cfg, num_workers=8, num_tasks=200)

        combined = {"N_res": N, **write_res, **read_res, **parallel_res}
        all_results.append(combined)

    df = pd.DataFrame(all_results)

    df["npz_write_MBps"] = df["npz_size"] / (df["npz_write"] * 1e6)
    df["hdf5_write_MBps"] = df["hdf5_size"] / (df["hdf5_write"] * 1e6)
    df["zarr_write_MBps"] = df["zarr_size"] / (df["zarr_write"] * 1e6)

    def useful_bytes(cfg):
        return 32 * 32 * 4  # float32

    # Partial Read Efficiency
    df["useful_bytes"] = useful_bytes(CONFIG)
    df["hdf5_efficiency"] = df["useful_bytes"] / df["hdf5_head_read"]
    df["zarr_efficiency"] = df["useful_bytes"] / df["zarr_head_read"]
    
    print("\n=== RESULTS ===")
    print(df)
    df.to_csv("benchmark_results.csv", index=False)


    # # =========================================================
    # # Write Performance (System Cost Baseline)
    # # =========================================================
    # plt.figure()
    # plt.plot(df["N_res"], df["hdf5_write"], marker="o", label="HDF5")
    # plt.plot(df["N_res"], df["zarr_write"], marker="o", label="Zarr")
    # plt.title("Write Performance vs Sequence Length")
    # plt.xlabel("Sequence Length (N_res)")
    # plt.ylabel("Time (seconds)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


    # =========================================================
    # Storage Footprint (Efficiency Baseline)
    # =========================================================
    plt.figure()
    plt.plot(df["N_res"], df["hdf5_size"], marker="o", label="HDF5")
    plt.plot(df["N_res"], df["zarr_size"], marker="o", label="Zarr")
    plt.plot(df["N_res"], df["npz_size"], marker="o", label="NPZ")
    plt.title("Storage Size Scaling")
    plt.xlabel("Sequence Length (N_res)")
    plt.ylabel("Bytes")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # =========================================================
    # Full-Archive Read (Not Primary Workload)
    # =========================================================
    plt.figure()
    plt.plot(df["N_res"], df["hdf5_full_read"], marker="o", label="HDF5")
    plt.plot(df["N_res"], df["zarr_full_read"], marker="o", label="Zarr")
    plt.plot(df["N_res"], df["npz_full_read"], marker="o", label="NPZ")
    plt.title("Full Archive Read Performance")
    plt.xlabel("Sequence Length (N_res)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # =========================================================
    # Partial Access Latency (Interpretability Core)
    # =========================================================
    plt.figure()
    plt.plot(df["N_res"], df["hdf5_layer_read"], marker="o", label="HDF5")
    plt.plot(df["N_res"], df["zarr_layer_read"], marker="o", label="Zarr")
    plt.title("Layer-Level Partial Read Latency")
    plt.xlabel("Sequence Length (N_res)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # # =========================================================
    # # Fine-Grained Access (Head-Level Querying)
    # # =========================================================
    # plt.figure()
    # plt.plot(df["N_res"], df["hdf5_head_read"], marker="o", label="HDF5")
    # plt.plot(df["N_res"], df["zarr_head_read"], marker="o", label="Zarr")
    # plt.title("Attention Head Access Latency")
    # plt.xlabel("Sequence Length (N_res)")
    # plt.ylabel("Time (seconds)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


    # =========================================================
    # Random Access Scaling (Key Interpretability Signal)
    # =========================================================
    plt.figure()
    plt.plot(df["N_res"], df["hdf5_random"], marker="o", label="HDF5")
    plt.plot(df["N_res"], df["zarr_random"], marker="o", label="Zarr")
    plt.title("Random Access Scaling (Interpretability Simulation)")
    plt.xlabel("Sequence Length (N_res)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # # =========================================================
    # # End-to-End Interpretability Workload (VizFold)
    # # =========================================================
    # plt.figure()
    # plt.plot(df["N_res"], df["viz_hdf5"], marker="o", label="HDF5")
    # plt.plot(df["N_res"], df["viz_zarr"], marker="o", label="Zarr")
    # plt.title("End-to-End Interpretability Workload (VizFold Simulation)")
    # plt.xlabel("Sequence Length (N_res)")
    # plt.ylabel("Time (seconds)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


    # # =========================================================
    # # Access Pattern Sensitivity
    # # =========================================================
    # plt.figure()
    # plt.plot(df["N_res"], df["hdf5_random"], linestyle="--", label="HDF5 (Random)")
    # plt.plot(df["N_res"], df["hdf5_layer_read"], linestyle="-", label="HDF5 (Layer)")
    # plt.plot(df["N_res"], df["zarr_random"], linestyle="--", label="Zarr (Random)")
    # plt.plot(df["N_res"], df["zarr_layer_read"], linestyle="-", label="Zarr (Layer)")

    # plt.title("Access Pattern Sensitivity Comparison")
    # plt.xlabel("Sequence Length (N_res)")
    # plt.ylabel("Time (seconds)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


    # =========================================================
    # Parallel Interpretability Workload
    # =========================================================
    plt.figure()
    plt.plot(df["N_res"], df["hdf5_parallel"], marker="o", label="HDF5")
    plt.plot(df["N_res"], df["zarr_parallel"], marker="o", label="Zarr")

    plt.title("Parallel Interpretability Workload")
    plt.xlabel("Sequence Length (N_res)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df


if __name__ == "__main__":
    run_benchmark()