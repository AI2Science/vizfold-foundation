import json
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

    # Component-separated output roots (PoC for modular pipeline reporting)
    COMP_ROOT = os.path.join(TRACE_DIR, "components")
    MSA_ATTN_DIR = os.path.join(COMP_ROOT, "msa", "attn_txt")
    PAIR_ATTN_DIR = os.path.join(COMP_ROOT, "pairformer_boltz", "attn_txt")
    SM_ATTN_DIR = os.path.join(COMP_ROOT, "sm_boltz", "attn_txt")
    os.makedirs(MSA_ATTN_DIR, exist_ok=True)
    os.makedirs(PAIR_ATTN_DIR, exist_ok=True)
    os.makedirs(SM_ATTN_DIR, exist_ok=True)

    ACT_DIR = os.environ.get("BOLTZ_ACT_DIR", "").strip()
    if ACT_DIR:
        os.makedirs(ACT_DIR, exist_ok=True)
    else:
        ACT_DIR = None
    PAIR_ACT_DIR = None
    if ACT_DIR is not None:
        PAIR_ACT_DIR = os.path.join(ACT_DIR, "pairformer_boltz")
        os.makedirs(PAIR_ACT_DIR, exist_ok=True)

    head_raw = os.environ.get("BOLTZ_TRACE_HEAD", "0").strip().lower()
    ALL_HEADS = head_raw in ("all", "-1")
    HEAD = None if ALL_HEADS else int(head_raw)

    TOPK = int(os.environ.get("BOLTZ_TRACE_TOPK", "50"))
    RESIDUES = [int(x) for x in os.environ.get("BOLTZ_TRACE_RESIDUES", "18").split(",") if x.strip()]
    LAYER_SET = {int(x) for x in os.environ.get("BOLTZ_TRACE_LAYERS", "0").split(",") if x.strip()}
    DEBUG = os.environ.get("BOLTZ_TRACE_DEBUG", "0") == "1"
    ENABLE_EXPERIMENTAL_MSA = os.environ.get("BOLTZ_TRACE_EXPERIMENTAL_MSA", "1") == "1"

    COMPONENT_STATUS = {
        "msa": {"available": False, "source": None, "files_written": 0},
        "pairformer_boltz": {"available": False, "source": None, "files_written": 0},
        "sm_boltz": {"available": False, "source": None, "files_written": 0},
    }

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

    def _note_write(component):
        ent = COMPONENT_STATUS.get(component)
        if ent is not None:
            ent["available"] = True
            ent["files_written"] += 1
            write_component_status()

    def _set_source(component, src):
        ent = COMPONENT_STATUS.get(component)
        if ent is not None and ent.get("source") is None:
            ent["source"] = src
            write_component_status()

    def _status_path():
        return os.path.join(TRACE_DIR, "component_status.json")

    def write_component_status():
        try:
            with open(_status_path(), "w") as f:
                json.dump(COMPONENT_STATUS, f, indent=2)
        except Exception as e:
            if DEBUG:
                print(f"[BOLTZ_TRACE] failed writing component_status.json: {e}", file=sys.stderr)

    def write_msa_row_heads(mats, layer_idx, out_dir=PAIR_ATTN_DIR):
        # mats: (H, N, N)
        Hh, N, _ = mats.shape
        out_path = os.path.join(out_dir, f"msa_row_attn_layer{layer_idx}.txt")
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

    def write_row_heads(mats, layer_idx, kind, out_dir=PAIR_ATTN_DIR):
        # mats: (H, N, N)
        Hh, N, _ = mats.shape
        for r in RESIDUES:
            if r < 0 or r >= N:
                continue

            out_path = os.path.join(out_dir, f"{kind}_attn_layer{layer_idx}_residue_idx_{r}.txt")
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
        if PAIR_ACT_DIR is None:
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
            path = os.path.join(PAIR_ACT_DIR, f"{kind}_act_layer{layer_idx}.npz")
            np.savez_compressed(path, pair_norm=pair_norm, pair_slice=pair_slice)
            _note_write("pairformer_boltz")
        except Exception as e:
            if DEBUG:
                print(f"[BOLTZ_TRACE] activation save failed layer={layer_idx} kind={kind}: {e}", file=sys.stderr)

    def install_hooks(model):
        from boltz.model.layers.triangular_attention.attention import TriangleAttention, TriangleAttentionEndingNode

        seen_start = set()
        seen_end = set()
        fired_start = set()
        fired_end = set()
        attached = 0

        def _first_tensor(x):
            try:
                import torch
            except Exception:
                return None
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x
            if isinstance(x, (tuple, list)):
                for it in x:
                    t = _first_tensor(it)
                    if t is not None:
                        return t
            if isinstance(x, dict):
                for it in x.values():
                    t = _first_tensor(it)
                    if t is not None:
                        return t
            return None

        def _iter_tensors(x):
            try:
                import torch
            except Exception:
                return
            if x is None:
                return
            if isinstance(x, torch.Tensor):
                yield x
                return
            if isinstance(x, (tuple, list)):
                for it in x:
                    yield from _iter_tensors(it)
                return
            if isinstance(x, dict):
                for it in x.values():
                    yield from _iter_tensors(it)
                return

        def _proxy_mats_from_out(out):
            """Fallback: derive a single-head (1,N,N) weight matrix from TriangleAttention output.

            Boltz versions may not expose real attention weights. However, TriangleAttention typically
            returns an updated pair representation of shape (N,N,C). We convert it to weights by
            taking the L2 norm across channels.
            """
            try:
                import torch
            except Exception:
                return None
            # TriangleAttention outputs can be nested structures; pick the best square (N,N,C)
            # (or (B,N,N,C)) tensor we can find.
            cand = None
            for t in _iter_tensors(out):
                if t is None:
                    continue
                if t.ndim == 4:
                    # expect (B,N,N,C)
                    if t.shape[0] >= 1 and t.shape[1] == t.shape[2]:
                        cand = t[0]
                        break
                elif t.ndim == 3:
                    if t.shape[0] == t.shape[1]:
                        cand = t
                        break
            if cand is None:
                return None
            t = cand
            if DEBUG:
                try:
                    print(f"[BOLTZ_TRACE] proxy source tensor shape={tuple(t.shape)} dtype={getattr(t,'dtype',None)}", file=sys.stderr)
                except Exception:
                    pass
            w = torch.linalg.vector_norm(t.float(), ord=2, dim=-1)  # (N,N)
            if not torch.isfinite(w).all():
                w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            # Normalize to avoid extreme value ranges (keeps top-k comparable across runs)
            try:
                denom = torch.max(w)
                if torch.isfinite(denom) and float(denom) > 0:
                    w = w / denom
            except Exception:
                pass
            mats = w.unsqueeze(0)  # (1,N,N)
            try:
                x = mats.detach()
                if getattr(x, "is_cuda", False):
                    x = x.cpu()
                x = x.float().numpy()
                if DEBUG:
                    try:
                        mn = float(np.min(x))
                        mx = float(np.max(x))
                        mean = float(np.mean(x))
                        print(f"[BOLTZ_TRACE] proxy mats stats min={mn:.6g} max={mx:.6g} mean={mean:.6g}", file=sys.stderr)
                    except Exception:
                        pass
                return x
            except Exception:
                return mats

        def _looks_like_attn(t):
            try:
                import torch
            except Exception:
                return False
            if not isinstance(t, torch.Tensor):
                return False
            if t.ndim < 3:
                return False
            try:
                return int(t.shape[-1]) == int(t.shape[-2])
            except Exception:
                return False

        def _extract_attn_from_output(out):
            try:
                import torch
            except Exception:
                torch = None
            if out is None:
                return None

            # If the module directly returns a tensor, it might already be attention.
            if torch is not None and isinstance(out, torch.Tensor):
                return out if _looks_like_attn(out) else None

            if isinstance(out, (tuple, list)):
                # Prefer tensors that look like attention.
                for item in out:
                    if torch is not None and isinstance(item, torch.Tensor) and _looks_like_attn(item):
                        return item
                # Otherwise recurse.
                for item in out:
                    if isinstance(item, (tuple, list, dict)):
                        a = _extract_attn_from_output(item)
                        if a is not None:
                            return a
                return None

            if isinstance(out, dict):
                for k in ('attn', 'attn_weights', 'attention', 'weights'):
                    v = out.get(k)
                    if v is not None:
                        if torch is not None and isinstance(v, torch.Tensor) and _looks_like_attn(v):
                            return v
                        a = _extract_attn_from_output(v)
                        if a is not None:
                            return a
                for v in out.values():
                    a = _extract_attn_from_output(v)
                    if a is not None:
                        return a
                return None

            return None

        for name, mod in model.named_modules():
            m = layer_pat.search(name)
            if not m:
                continue
            layer_idx = int(m.group(1))
            if layer_idx not in LAYER_SET:
                continue

            if isinstance(mod, TriangleAttentionEndingNode):
                def _hook_end(mmod, inp, out, _layer=layer_idx):
                    if DEBUG and _layer not in fired_end:
                        fired_end.add(_layer)
                        print(f"[BOLTZ_TRACE] hook_end fired layer={_layer} type={type(mmod).__name__}", file=sys.stderr)
                    if _layer in seen_end:
                        return
                    if not hasattr(mmod, "mha"):
                        if DEBUG:
                            print(f"[BOLTZ_TRACE] hook_end missing mha layer={_layer}", file=sys.stderr)
                        return
                    mha = mmod.mha
                    la = getattr(mha, "_last_attn", None)
                    if la is None:
                        la = getattr(mha, "_vizfold_last_attn", None)
                    src = "true_attention"
                    if la is None:
                        mats = _proxy_mats_from_out(out)
                        if mats is None:
                            mats = _proxy_mats_from_out(getattr(mha, "_vizfold_last_out", None))
                        if mats is None:
                            if DEBUG:
                                print(f"[BOLTZ_TRACE] hook_end no attn; proxy failed layer={_layer}", file=sys.stderr)
                            return
                        if DEBUG:
                            print(f"[BOLTZ_TRACE] hook_end using proxy weights layer={_layer}", file=sys.stderr)
                        src = "proxy_pair_norm"
                    else:
                        mats = mats_from_last_attn(la)
                        if mats is None:
                            if DEBUG:
                                try:
                                    shp = tuple(getattr(la, 'shape', ()))
                                except Exception:
                                    shp = None
                                print(f"[BOLTZ_TRACE] hook_end mats_from_last_attn returned None layer={_layer} last_attn_shape={shp}", file=sys.stderr)
                            return

                    # choose heads
                    if HEAD is None:
                        use = mats
                    else:
                        if HEAD < 0 or HEAD >= mats.shape[0]:
                            return
                        use = mats[HEAD:HEAD+1]

                    # Writers expect numpy arrays (they call .copy())
                    try:
                        import torch
                        if isinstance(use, torch.Tensor):
                            u = use.detach()
                            if getattr(u, "is_cuda", False):
                                u = u.cpu()
                            use = u.float().numpy()
                    except Exception:
                        pass

                    seen_end.add(_layer)
                    try:
                        write_row_heads(use, _layer, "triangle_end", out_dir=PAIR_ATTN_DIR)
                        _note_write("pairformer_boltz")
                        _set_source("pairformer_boltz", src)
                    except Exception as e:
                        if DEBUG:
                            print(f"[BOLTZ_TRACE] write_row_heads failed kind=triangle_end layer={_layer}: {e}", file=sys.stderr)
                    save_activation_npz("triangle_end", _layer, out)

                mod.register_forward_hook(_hook_end)
                attached += 1

            elif isinstance(mod, TriangleAttention) and getattr(mod, "starting", True):
                def _hook_start(mmod, inp, out, _layer=layer_idx):
                    if DEBUG and _layer not in fired_start:
                        fired_start.add(_layer)
                        print(f"[BOLTZ_TRACE] hook_start fired layer={_layer} type={type(mmod).__name__}", file=sys.stderr)
                    if _layer in seen_start:
                        return
                    if not hasattr(mmod, "mha"):
                        if DEBUG:
                            print(f"[BOLTZ_TRACE] hook_start missing mha layer={_layer}", file=sys.stderr)
                        return
                    mha = mmod.mha
                    la = getattr(mha, "_last_attn", None)
                    if la is None:
                        la = getattr(mha, "_vizfold_last_attn", None)
                    src = "true_attention"
                    if la is None:
                        mats = _proxy_mats_from_out(out)
                        if mats is None:
                            mats = _proxy_mats_from_out(getattr(mha, "_vizfold_last_out", None))
                        if mats is None:
                            if DEBUG:
                                print(f"[BOLTZ_TRACE] hook_start no attn; proxy failed layer={_layer}", file=sys.stderr)
                            return
                        if DEBUG:
                            print(f"[BOLTZ_TRACE] hook_start using proxy weights layer={_layer}", file=sys.stderr)
                        src = "proxy_pair_norm"
                    else:
                        mats = mats_from_last_attn(la)
                        if mats is None:
                            if DEBUG:
                                try:
                                    shp = tuple(getattr(la, 'shape', ()))
                                except Exception:
                                    shp = None
                                print(f"[BOLTZ_TRACE] hook_start mats_from_last_attn returned None layer={_layer} last_attn_shape={shp}", file=sys.stderr)
                            return

                    if HEAD is None:
                        use = mats
                    else:
                        if HEAD < 0 or HEAD >= mats.shape[0]:
                            return
                        use = mats[HEAD:HEAD+1]

                    # Writers expect numpy arrays (they call .copy())
                    try:
                        import torch
                        if isinstance(use, torch.Tensor):
                            u = use.detach()
                            if getattr(u, "is_cuda", False):
                                u = u.cpu()
                            use = u.float().numpy()
                    except Exception:
                        pass

                    seen_start.add(_layer)
                    try:
                        # In Boltz PoC, msa_row is produced from pairformer/triangle stream.
                        # Write this explicitly under pairformer_boltz outputs.
                        write_msa_row_heads(use, _layer, out_dir=PAIR_ATTN_DIR)
                        _note_write("pairformer_boltz")
                        _set_source("pairformer_boltz", src)
                    except Exception as e:
                        if DEBUG:
                            print(f"[BOLTZ_TRACE] write_msa_row_heads failed layer={_layer}: {e}", file=sys.stderr)
                    try:
                        write_row_heads(use, _layer, "triangle_start", out_dir=PAIR_ATTN_DIR)
                        _note_write("pairformer_boltz")
                        _set_source("pairformer_boltz", src)
                    except Exception as e:
                        if DEBUG:
                            print(f"[BOLTZ_TRACE] write_row_heads failed kind=triangle_start layer={_layer}: {e}", file=sys.stderr)
                    save_activation_npz("triangle_start", _layer, out)

                mod.register_forward_hook(_hook_start)
                attached += 1

            # Experimental best-effort MSA hook discovery (may not fire in all Boltz versions)
            if ENABLE_EXPERIMENTAL_MSA:
                lname = name.lower()
                if "msa" in lname and hasattr(mod, "mha"):
                    def _hook_msa(mmod, inp, out, _layer=layer_idx):
                        mha = getattr(mmod, "mha", None)
                        la = getattr(mha, "_last_attn", None) if mha is not None else None
                        if la is None and mha is not None:
                            la = getattr(mha, "_vizfold_last_attn", None)
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
                        try:
                            import torch
                            if isinstance(use, torch.Tensor):
                                u = use.detach()
                                if getattr(u, "is_cuda", False):
                                    u = u.cpu()
                                use = u.float().numpy()
                        except Exception:
                            pass
                        try:
                            write_msa_row_heads(use, _layer, out_dir=MSA_ATTN_DIR)
                            _note_write("msa")
                            _set_source("msa", "experimental_true_attention")
                        except Exception:
                            return
                    mod.register_forward_hook(_hook_msa)
                    attached += 1

            if hasattr(mod, "mha"):
                try:
                    mha = mod.mha
                    if mha is not None and not getattr(mha, "_vizfold_capture_installed", False):
                        def _mha_hook(mmod, inp, out):
                            # One-time debug dump to understand what boltz returns.
                            if DEBUG and not getattr(mmod, "_vizfold_debug_dumped", False):
                                try:
                                    import torch
                                    def _summ(x):
                                        if isinstance(x, torch.Tensor):
                                            return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype})"
                                        return type(x).__name__
                                    if isinstance(out, (tuple, list)):
                                        print(f"[BOLTZ_TRACE] mha out is {type(out).__name__} len={len(out)}", file=sys.stderr)
                                        for i, it in enumerate(out[:6]):
                                            print(f"[BOLTZ_TRACE]   out[{i}]={_summ(it)}", file=sys.stderr)
                                    elif isinstance(out, dict):
                                        keys = list(out.keys())
                                        print(f"[BOLTZ_TRACE] mha out is dict keys={keys[:12]}", file=sys.stderr)
                                    else:
                                        print(f"[BOLTZ_TRACE] mha out is {_summ(out)}", file=sys.stderr)
                                    # Also inspect common attribute names
                                    for attr in ("attn", "attn_weights", "attention", "weights", "_attn", "_attn_weights"):
                                        if hasattr(mmod, attr):
                                            v = getattr(mmod, attr)
                                            if isinstance(v, torch.Tensor):
                                                print(f"[BOLTZ_TRACE] mha has attr {attr}=Tensor(shape={tuple(v.shape)})", file=sys.stderr)
                                except Exception as e:
                                    print(f"[BOLTZ_TRACE] mha debug dump failed: {e}", file=sys.stderr)
                                mmod._vizfold_debug_dumped = True

                            attn = _extract_attn_from_output(out)
                            # Always cache the output for proxy fallbacks (some versions never expose true weights).
                            try:
                                if getattr(mmod, "_vizfold_last_out", None) is None:
                                    setattr(mmod, "_vizfold_last_out", out)
                            except Exception:
                                pass
                            if attn is None:
                                # Fallback: check for common attribute names if module stores attention.
                                for attr in ("attn", "attn_weights", "attention", "weights", "_attn", "_attn_weights"):
                                    try:
                                        v = getattr(mmod, attr)
                                    except Exception:
                                        v = None
                                    if v is not None and _looks_like_attn(v):
                                        attn = v
                                        break

                            if attn is not None:
                                setattr(mmod, "_vizfold_last_attn", attn)
                        mha.register_forward_hook(_mha_hook)
                        mha._vizfold_capture_installed = True
                        if DEBUG:
                            print(f"[BOLTZ_TRACE] installed mha capture hook on {type(mha).__name__}", file=sys.stderr)
                except Exception as e:
                    if DEBUG:
                        print(f"[BOLTZ_TRACE] failed installing mha capture hook: {e}", file=sys.stderr)

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
            write_component_status()
            return model

        Boltz2.load_from_checkpoint = classmethod(_new_lfc)

        if DEBUG:
            print("[BOLTZ_TRACE] patched Boltz2.load_from_checkpoint", file=sys.stderr)
    except Exception as e:
        print("[BOLTZ_TRACE] failed to patch Boltz2.load_from_checkpoint:", e, file=sys.stderr)
