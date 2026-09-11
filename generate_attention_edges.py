import argparse
import os
import pickle
import numpy as np


def find_attention_pkl(attn_map_dir: str, tag: str) -> str:
    """
    Try a couple of reasonable filename patterns to locate the attention map.
    Falls back to scanning the directory for a file that starts with the tag.
    """
    candidates = [
        os.path.join(attn_map_dir, f"{tag}_attention.pkl"),
        os.path.join(attn_map_dir, f"{tag}.pkl"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    # Fallback - scan directory
    for fname in os.listdir(attn_map_dir):
        if fname.startswith(tag) and fname.endswith(".pkl"):
            return os.path.join(attn_map_dir, fname)

    raise FileNotFoundError(
        f"Could not find attention pickle for tag '{tag}' in '{attn_map_dir}'."
    )


def extract_topk_from_matrix(mat: np.ndarray, top_k: int, drop_diagonal: bool) -> np.ndarray:
    """
    Given an [N, N] attention matrix, return an array of shape [K, 3]
    with (i, j, score) sorted by score descending.
    """
    n = mat.shape[0]
    scores = mat.copy()

    if drop_diagonal:
        # Ignore self attention
        np.fill_diagonal(scores, -np.inf)

    flat = scores.reshape(-1)
    k = min(top_k, flat.size)

    if k <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Get top k indices
    top_idx = np.argpartition(flat, -k)[-k:]
    # Sort those top k by score descending
    top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]

    rows = top_idx // n
    cols = top_idx % n
    vals = flat[top_idx]

    return np.stack([rows, cols, vals], axis=-1)


def dump_edges_for_type(
    attn_tensor: np.ndarray,
    attn_type: str,
    tag: str,
    out_dir: str,
    top_k: int,
    drop_diagonal: bool,
) -> None:
    """
    attn_tensor shape - [L, H, N, N]
    Writes one TSV per (layer, head) and one combined TSV per attention type.
    """
    os.makedirs(out_dir, exist_ok=True)

    num_layers, num_heads, n, _ = attn_tensor.shape

    combined_rows = []

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            mat = attn_tensor[layer_idx, head_idx]

            top_entries = extract_topk_from_matrix(
                mat, top_k=top_k, drop_diagonal=drop_diagonal
            )

            if top_entries.shape[0] == 0:
                continue

            # Save per layer head TSV
            per_file = os.path.join(
                out_dir,
                f"{tag}_{attn_type}_layer{layer_idx}_head{head_idx}_top{top_k}.tsv",
            )

            with open(per_file, "w") as f:
                f.write("layer\thead\tsrc_idx\tdst_idx\tscore\n")
                for i, j, score in top_entries:
                    f.write(
                        f"{layer_idx}\t{head_idx}\t{int(i)}\t{int(j)}\t{float(score)}\n"
                    )

            # Also accumulate into combined table
            for i, j, score in top_entries:
                combined_rows.append(
                    (
                        attn_type,
                        layer_idx,
                        head_idx,
                        int(i),
                        int(j),
                        float(score),
                    )
                )

    if combined_rows:
        combined_path = os.path.join(
            out_dir, f"{tag}_{attn_type}_all_layers_heads_top{top_k}.tsv"
        )
        with open(combined_path, "w") as f:
            f.write("attn_type\tlayer\thead\tsrc_idx\tdst_idx\tscore\n")
            for attn, layer_idx, head_idx, i, j, score in combined_rows:
                f.write(
                    f"{attn}\t{layer_idx}\t{head_idx}\t{i}\t{j}\t{score}\n"
                )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    attn_pkl_path = find_attention_pkl(args.attn_map_dir, args.tag)
    print(f"[info] Loading attention pickle from {attn_pkl_path}")
    with open(attn_pkl_path, "rb") as f:
        viz_data = pickle.load(f)

    # Expected keys from generate_viz_data.py
    #   - "msa_row_attention": [L, H, N, N]
    #   - "msa_col_attention": [L, H, N, N]
    #   - "pair_attention": [L, H, N, N]
    attn_types = args.attn_types
    if attn_types is None or len(attn_types) == 0:
        # Default to whatever keys are present
        attn_types = list(viz_data.keys())

    for attn_type in attn_types:
        if attn_type not in viz_data:
            print(f"[warn] Attention type '{attn_type}' not present in pickle, skipping")
            continue

        print(f"[info] Processing '{attn_type}'")
        tensor = np.asarray(viz_data[attn_type], dtype=np.float32)

        out_dir = os.path.join(args.output_dir, attn_type)
        dump_edges_for_type(
            tensor,
            attn_type=attn_type,
            tag=args.tag,
            out_dir=out_dir,
            top_k=args.top_k,
            drop_diagonal=args.drop_diagonal,
        )

    print(f"[done] Edge lists written under {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert saved attention pickles into top k residue pair edge lists "
            "for downstream visualization."
        )
    )
    parser.add_argument(
        "attn_map_dir",
        type=str,
        help="Directory containing <tag>_attention.pkl produced by generate_viz_data.py",
    )
    parser.add_argument(
        "tag",
        type=str,
        help="Tag used when saving the attention pickle - typically derived from the FASTA header",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="attention_edges",
        help="Directory where TSV files with top k edges will be written",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Number of highest scoring residue pairs to keep per layer and head",
    )
    parser.add_argument(
        "--attn_types",
        nargs="*",
        default=["msa_row_attention", "msa_col_attention", "pair_attention"],
        help=(
            "Attention types to process from the pickle - "
            "defaults to msa_row_attention, msa_col_attention, pair_attention"
        ),
    )
    parser.add_argument(
        "--drop_diagonal",
        action="store_true",
        default=False,
        help="If set, ignore self attention entries where src_idx equals dst_idx",
    )

    main(parser.parse_args())
