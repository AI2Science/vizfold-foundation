from __future__ import annotations

import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except Exception:
    torch = None


ArrayLike = Union[np.ndarray, "torch.Tensor"]


@dataclass
class EvoformerRunArtifact:
    run_dir: Union[str, Path]
    attention_dir: Optional[Union[str, Path]] = None
    output_pkl: Optional[Union[str, Path]] = None
    reps_path: Optional[Union[str, Path]] = None
    output_dict: Optional[Dict[str, Any]] = field(default=None, init=False)
    reps: Optional[Dict[str, Any]] = field(default=None, init=False)

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        self.attention_dir = Path(self.attention_dir) if self.attention_dir else None
        self.output_pkl = Path(self.output_pkl) if self.output_pkl else None
        self.reps_path = Path(self.reps_path) if self.reps_path else None

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        if torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _save_or_show(fig, save_path=None, show=True, dpi=200):
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def load_output_dict(self):
        if self.output_pkl is None:
            candidates = list((self.run_dir / "predictions").glob("*_output_dict.pkl"))
            if not candidates:
                raise FileNotFoundError("No output_dict.pkl found")
            self.output_pkl = candidates[0]
        with open(self.output_pkl, "rb") as f:
            self.output_dict = pickle.load(f)
        return self.output_dict

    def load_reps(self):
        if self.reps_path is None:
            raise FileNotFoundError("No reps_path provided")
        if torch is None:
            raise ImportError("torch is required to load .pt files")
        self.reps = torch.load(self.reps_path, map_location="cpu")
        return self.reps

    @staticmethod
    def load_attention_file_to_mats(path: str | Path):
        path = Path(path)
        header_re = re.compile(r"Layer\s+(\d+),\s+Head\s+(\d+)")
        head_triples = defaultdict(list)
        layer_id = None
        current_head = None
        max_idx = -1

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = header_re.match(line)
                if m:
                    layer_id = int(m.group(1))
                    current_head = int(m.group(2))
                    continue
                parts = line.split()
                if len(parts) != 3:
                    continue
                i, j, score = int(parts[0]), int(parts[1]), float(parts[2])
                head_triples[current_head].append((i, j, score))
                max_idx = max(max_idx, i, j)

        if layer_id is None:
            raise ValueError(f"Could not parse {path}")

        mats = {}
        size = max_idx + 1
        for head, triples in head_triples.items():
            mat = np.zeros((size, size), dtype=np.float32)
            for i, j, score in triples:
                mat[i, j] = score
            mats[head] = mat
        return layer_id, mats

    def attention_file(self, kind: str, layer: int, residue_idx: Optional[int] = None) -> Path:
        if self.attention_dir is None:
            raise FileNotFoundError("No attention_dir provided")
        if kind == "msa_row_attn":
            return self.attention_dir / f"msa_row_attn_layer{layer}.txt"
        if kind == "triangle_start_attn":
            if residue_idx is None:
                raise ValueError("triangle_start_attn requires residue_idx")
            return self.attention_dir / f"triangle_start_attn_layer{layer}_residue_idx_{residue_idx}.txt"
        raise ValueError(f"Unsupported kind: {kind}")

    def get_attention_matrix(self, kind: str, layer: int, head: int = 0, residue_idx: Optional[int] = None, mean_across_heads: bool = False):
        path = self.attention_file(kind, layer, residue_idx)
        _, mats = self.load_attention_file_to_mats(path)
        if mean_across_heads:
            return np.mean(np.stack([m for _, m in sorted(mats.items())], axis=0), axis=0)
        return mats[head]

    @staticmethod
    def low_rank_approximation(mat: ArrayLike, rank: int = 5):
        mat = EvoformerRunArtifact._to_numpy(mat)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        approx = (U[:, :rank] * S[:rank]) @ Vt[:rank, :]
        residual = mat - approx
        return approx, residual, S

    def plot_heatmap(self, mat: ArrayLike, title="", xlabel="Index j", ylabel="Index i", cmap="viridis", figsize=(6, 5), colorbar_label="Value", save_path=None, show=True):
        mat = self._to_numpy(mat)
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        fig.colorbar(im, ax=ax, label=colorbar_label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        self._save_or_show(fig, save_path=save_path, show=show)

    def plot_line(self, values: ArrayLike, title="", xlabel="Index", ylabel="Value", figsize=(8, 3), save_path=None, show=True):
        values = self._to_numpy(values)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(values)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save_or_show(fig, save_path=save_path, show=show)

    def plot_low_rank_diagnostics(self, mat: ArrayLike, rank: int = 5, title_prefix="Matrix", save_path=None, show=True):
        mat = self._to_numpy(mat)
        approx, residual, _ = self.low_rank_approximation(mat, rank=rank)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        im0 = axes[0].imshow(mat, aspect="auto", cmap="viridis")
        axes[0].set_title(f"{title_prefix} (original)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(approx, aspect="auto", cmap="viridis")
        axes[1].set_title(f"Rank-{rank} approximation")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(residual, aspect="auto", cmap="viridis")
        axes[2].set_title("Residual")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.tight_layout()
        self._save_or_show(fig, save_path=save_path, show=show)

    def plot_singular_values(self, mat: ArrayLike, title="Singular value spectrum", save_path=None, show=True):
        mat = self._to_numpy(mat)
        S = np.linalg.svd(mat, full_matrices=False, compute_uv=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(1, len(S) + 1), S, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular value")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save_or_show(fig, save_path=save_path, show=show)
        return S

    def plot_eigenvalue_histogram(self, mat: ArrayLike, title="Eigenvalue histogram", bins: int = 40, save_path=None, show=True):
        mat = self._to_numpy(mat)
        vals = np.real(np.linalg.eigvals(mat))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(vals, bins=bins)
        ax.set_title(title)
        ax.set_xlabel("Eigenvalue (real part)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        self._save_or_show(fig, save_path=save_path, show=show)
        return vals

    def plot_correlation_heatmap(self, features: ArrayLike, title="Correlation heatmap", save_path=None, show=True):
        features = self._to_numpy(features)
        corr = np.corrcoef(features)
        self.plot_heatmap(corr, title=title, xlabel="Index", ylabel="Index", cmap="coolwarm", colorbar_label="Correlation", save_path=save_path, show=show)
        return corr

    def make_report_style_figure(self, mat: ArrayLike, name="Matrix", rank: int = 5, save_path=None, show=True):
        mat = self._to_numpy(mat)
        _, residual, S = self.low_rank_approximation(mat, rank=rank)
        vals = np.real(np.linalg.eigvals(mat))

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        im0 = axes[0, 0].imshow(mat, aspect="auto", cmap="viridis")
        axes[0, 0].set_title(name)
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(residual, aspect="auto", cmap="viridis")
        axes[0, 1].set_title(f"Residual after rank-{rank}")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        axes[1, 0].plot(np.arange(1, len(S) + 1), S, marker="o")
        axes[1, 0].set_title("Singular values")
        axes[1, 0].set_xlabel("Index")
        axes[1, 0].set_ylabel("Value")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(vals, bins=40)
        axes[1, 1].set_title("Eigenvalue histogram")
        axes[1, 1].set_xlabel("Eigenvalue (real part)")
        axes[1, 1].set_ylabel("Count")

        fig.tight_layout()
        self._save_or_show(fig, save_path=save_path, show=show)

    def plot_attention_report(self, kind: str, layer: int, head: int = 0, residue_idx: Optional[int] = None, mean_across_heads: bool = False, rank: int = 5, save_path=None, show=True):
        mat = self.get_attention_matrix(kind, layer, head=head, residue_idx=residue_idx, mean_across_heads=mean_across_heads)
        label = f"{kind} - layer {layer}"
        if residue_idx is not None:
            label += f" - residue {residue_idx}"
        label += " - mean heads" if mean_across_heads else f" - head {head}"
        self.make_report_style_figure(mat, name=label, rank=rank, save_path=save_path, show=show)
        return mat

    def plot_pair_mean(self, pair: Optional[ArrayLike] = None, save_path=None, show=True):
        if pair is None:
            if self.reps is None:
                self.load_reps()
            pair = self.reps["pair"]
        pair = self._to_numpy(pair)
        pair_mean = pair.mean(axis=-1)
        self.plot_heatmap(pair_mean, title="Pair representation mean over channels", xlabel="Residue j", ylabel="Residue i", colorbar_label="Mean activation", save_path=save_path, show=show)
        return pair_mean

    def plot_pair_norm(self, pair: Optional[ArrayLike] = None, save_path=None, show=True):
        if pair is None:
            if self.reps is None:
                self.load_reps()
            pair = self.reps["pair"]
        pair = self._to_numpy(pair)
        pair_norm = np.linalg.norm(pair, axis=-1)
        self.plot_heatmap(pair_norm, title="Pair representation norm heatmap", xlabel="Residue j", ylabel="Residue i", colorbar_label="L2 norm", save_path=save_path, show=show)
        return pair_norm

    def plot_msa_mean_channels(self, msa: Optional[ArrayLike] = None, save_path=None, show=True):
        if msa is None:
            if self.reps is None:
                self.load_reps()
            msa = self.reps["msa"]
        msa = self._to_numpy(msa)
        msa_seq_mean = msa.mean(axis=0)
        self.plot_heatmap(msa_seq_mean.T, title="MSA representation (mean over sequences)", xlabel="Residue index", ylabel="Channel", colorbar_label="Activation", save_path=save_path, show=show)
        return msa_seq_mean

    def plot_msa_residue_norm(self, msa: Optional[ArrayLike] = None, save_path=None, show=True):
        if msa is None:
            if self.reps is None:
                self.load_reps()
            msa = self.reps["msa"]
        msa = self._to_numpy(msa)
        msa_seq_mean = msa.mean(axis=0)
        norms = np.linalg.norm(msa_seq_mean, axis=-1)
        self.plot_line(norms, title="MSA representation norm by residue", xlabel="Residue index", ylabel="L2 norm", save_path=save_path, show=show)
        return norms

    def save_standard_attention_bundle(self, out_dir: str | Path, residue_idx: int, layers: Iterable[int] = (0, 12, 24, 47), rank: int = 5, mean_across_heads: bool = True):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for layer in layers:
            self.plot_attention_report("msa_row_attn", layer=layer, mean_across_heads=mean_across_heads, rank=rank, save_path=out_dir / f"msa_row_attn_layer{layer}.png", show=False)
            self.plot_attention_report("triangle_start_attn", layer=layer, residue_idx=residue_idx, mean_across_heads=mean_across_heads, rank=rank, save_path=out_dir / f"triangle_start_attn_layer{layer}_residue_idx_{residue_idx}.png", show=False)

    def save_rep_bundle(self, out_dir: str | Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pair_mean = self.plot_pair_mean(save_path=out_dir / "pair_mean.png", show=False)
        self.make_report_style_figure(pair_mean, name="Pair representation mean", save_path=out_dir / "pair_mean_report.png", show=False)
        pair_norm = self.plot_pair_norm(save_path=out_dir / "pair_norm.png", show=False)
        self.make_report_style_figure(pair_norm, name="Pair representation norm", save_path=out_dir / "pair_norm_report.png", show=False)
        msa_mean = self.plot_msa_mean_channels(save_path=out_dir / "msa_mean_channels.png", show=False)
        self.plot_correlation_heatmap(msa_mean, title="Residue correlation from MSA rep", save_path=out_dir / "msa_residue_corr.png", show=False)
        self.plot_msa_residue_norm(save_path=out_dir / "msa_residue_norm.png", show=False)
