import os, re, sys

TRACE_DIR = os.environ.get("BOLTZ_TRACE_DIR")
if not TRACE_DIR:
    TRACE_ENABLED = False
else:
    TRACE_ENABLED = True

if not TRACE_ENABLED:
    pass
else:
    import numpy as np
    os.makedirs(TRACE_DIR, exist_ok=True)

    ACT_DIR = os.environ.get("BOLTZ_ACT_DIR", "").strip()
    if ACT_DIR:
        os.makedirs(ACT_DIR, exist_ok=True)
    else:
        ACT_DIR = None

    head_raw = os.environ.get("BOLTZ_TRACE_HEAD", "0").strip().lower()
    ALL_HEADS = head_raw in ("all", "-1")
    HEAD = None if ALL_HEADS else int(head_raw)

    TOPK = int(os.environ.get("BOLTZ_TRACE_TOPK", "50"))
    RESIDUES = [int(x) for x in os.environ.get("BOLTZ_TRACE_RESIDUES", "18").split(",") if x.strip()]
    LAYER_SET = {int(x) for x in os.environ.get("BOLTZ_TRACE_LAYERS", "0").split(",") if x.strip()}
    DEBUG = os.environ.get("BOLTZ_TRACE_DEBUG", "0") == "1"

    layer_pat = re.compile(r"layers\.(\d+)")

    def mats_from_last_attn(last_attn):
        """
        Normalize last_attn to numpy array (H, N, N).
        Handles common shapes:
          - (B, I, H, N, N) or (B, H, I, N, N)
          - (B, H, N, N)
          - (H, N, N)
          - (N, N)
        """
        a = last_attn.detach()
        if getattr(a, "is_cuda", False):
            a = a.cpu()
        a = a.float().numpy()

        # Drop batch dim if present
        if a.ndim >= 4:
            a = a[0]

        # (I, H, N, N) OR (H, I, N, N) -> mean over I -> (H, N, N)
        if a.ndim == 4:
            d0, d1, n2, n3 = a.shape
            if n2 != n3:
                if DEBUG:
                    print(f"[BOLTZ_TRACE] unexpected 4D attn shape {a.shape}", file=sys.stderr)
                return None

            # heuristic: head dim small, I dim larger
            if d0 <= 32 and d1 > 32:
                # (H, I, N, N)
                return a.mean(axis=1)
            else:
                # assume (I, H, N, N)
                return a.mean(axis=0)

        # (H, N, N)
        if a.ndim == 3:
            if a.shape[1] == a.shape[2]:
                return a
            if DEBUG:
                print(f"[BOLTZ_TRACE] unexpected 3D attn shape {a.shape}", file=sys.stderr)
            return None

        # (N, N) -> single head
        if a.ndim == 2:
            return a[None, :, :]

        if DEBUG:
            print(f"[BOLTZ_TRACE] unexpected attn shape {a.shape}", file=sys.stderr)
        return None

    def write_msa_row_heads(mats, layer_idx):
        # mats: (H, N, N)
        Hh, N, _ = mats.shape
        out_path = os.path.join(TRACE_DIR, f"msa_row_attn_layer{layer_idx}.txt")
        with open(out_path, "w") as f:
            for h in range(Hh):
                m = mats[h].copy()
                np.fill_diagonal(m, 0.0)

                iu = np.triu_indices(N, k=1)
                vals = m[iu]
                if vals.size == 0:
                    continue

                k = min(TOPK, vals.size)
                idx = np.argpartition(vals, -k)[-k:]
                edges = sorted(
                    [(int(iu[0][t]), int(iu[1][t]), float(vals[t])) for t in idx],
                    key=lambda x: x[2],
                    reverse=True,
                )

                f.write(f"Layer {layer_idx} Head {h}\n")
                for i, j, v in edges:
                    f.write(f"{i} {j} {v}\n")

    def write_row_heads(mats, layer_idx, kind):
        # mats: (H, N, N)
        Hh, N, _ = mats.shape
        for r in RESIDUES:
            if r < 0 or r >= N:
                continue

            out_path = os.path.join(TRACE_DIR, f"{kind}_attn_layer{layer_idx}_residue_idx_{r}.txt")
            with open(out_path, "w") as f:
                for h in range(Hh):
                    row = mats[h, r].copy()
                    row[r] = 0.0

                    k = min(TOPK, max(0, row.size - 1))
                    if k == 0:
                        continue

                    js = np.argpartition(row, -k)[-k:]
                    edges = sorted(
                        [(r, int(j), float(row[j])) for j in js if j != r],
                        key=lambda x: x[2],
                        reverse=True,
                    )

                    f.write(f"Layer {layer_idx} Head {h}\n")
                    for i, j, v in edges:
                        f.write(f"{i} {j} {v}\n")

    def save_activation_npz(kind, layer_idx, out_tensor):
        """
        Save lightweight activation:
          - pair_norm: (N,N) norm over channels
          - pair_slice: (N,N,8) first 8 channels
        """
        if ACT_DIR is None:
            return
        try:
            import torch
            if not isinstance(out_tensor, torch.Tensor):
                return
            if out_tensor.dim() != 4:
                return
            # expect (B, N, N, C)
            x = out_tensor.detach()
            if x.is_cuda:
                x = x.cpu()
            x = x.float()[0]  # (N,N,C)
            pair_norm = torch.linalg.vector_norm(x, dim=-1).numpy().astype("float16")
            pair_slice = x[..., :8].numpy().astype("float16")
            path = os.path.join(ACT_DIR, f"{kind}_act_layer{layer_idx}.npz")
            np.savez_compressed(path, pair_norm=pair_norm, pair_slice=pair_slice)
        except Exception as e:
            if DEBUG:
                print(f"[BOLTZ_TRACE] activation save failed layer={layer_idx} kind={kind}: {e}", file=sys.stderr)

    def install_hooks(model):
        from boltz.model.layers.triangular_attention.attention import TriangleAttention, TriangleAttentionEndingNode

        seen_start = set()
        seen_end = set()
        attached = 0

        for name, mod in model.named_modules():
            m = layer_pat.search(name)
            if not m:
                continue
            layer_idx = int(m.group(1))
            if layer_idx not in LAYER_SET:
                continue

            if isinstance(mod, TriangleAttentionEndingNode):
                def _hook_end(mmod, inp, out, _layer=layer_idx):
                    if _layer in seen_end:
                        return
                    if not hasattr(mmod, "mha") or not hasattr(mmod.mha, "_last_attn"):
                        return
                    la = mmod.mha._last_attn
                    if la is None:
                        return
                    mats = mats_from_last_attn(la)
                    if mats is None:
                        return

                    # choose heads
                    if HEAD is None:
                        use = mats
                    else:
                        if HEAD < 0 or HEAD >= mats.shape[0]:
                            return
                        use = mats[HEAD:HEAD+1]

                    seen_end.add(_layer)
                    write_row_heads(use, _layer, "triangle_end")
                    save_activation_npz("triangle_end", _layer, out)

                mod.register_forward_hook(_hook_end)
                attached += 1

            elif isinstance(mod, TriangleAttention) and getattr(mod, "starting", True):
                def _hook_start(mmod, inp, out, _layer=layer_idx):
                    if _layer in seen_start:
                        return
                    if not hasattr(mmod, "mha") or not hasattr(mmod.mha, "_last_attn"):
                        return
                    la = mmod.mha._last_attn
                    if la is None:
                        return
                    mats = mats_from_last_attn(la)
                    if mats is None:
                        return

                    if HEAD is None:
                        use = mats
                    else:
                        if HEAD < 0 or HEAD >= mats.shape[0]:
                            return
                        use = mats[HEAD:HEAD+1]

                    seen_start.add(_layer)
                    write_msa_row_heads(use, _layer)
                    write_row_heads(use, _layer, "triangle_start")
                    save_activation_npz("triangle_start", _layer, out)

                mod.register_forward_hook(_hook_start)
                attached += 1

        if DEBUG:
            print(f"[BOLTZ_TRACE] attached {attached} hooks", file=sys.stderr)

    # Patch Boltz2.load_from_checkpoint
    try:
        from boltz.model.models.boltz2 import Boltz2
        _orig = Boltz2.load_from_checkpoint

        def _new_lfc(cls, *args, **kwargs):
            # works whether _orig is function or classmethod
            try:
                model = _orig(*args, **kwargs)
            except TypeError:
                model = _orig(cls, *args, **kwargs)
            if not getattr(model, "_vizfold_hooks_installed", False):
                install_hooks(model)
                model._vizfold_hooks_installed = True
            return model

        Boltz2.load_from_checkpoint = classmethod(_new_lfc)

        if DEBUG:
            print("[BOLTZ_TRACE] patched Boltz2.load_from_checkpoint", file=sys.stderr)
    except Exception as e:
        print("[BOLTZ_TRACE] failed to patch Boltz2.load_from_checkpoint:", e, file=sys.stderr)
