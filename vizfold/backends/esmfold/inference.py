"""
ESMFold inference: load model, run forward, optionally extract traces.

Uses HuggingFace Transformers (EsmForProteinFolding) to avoid OpenFold build dependency.
"""
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch

from vizfold.backends.base import BackendBase
from vizfold.backends.esmfold.hooks import ESMFoldTraceCollector, StructureModuleTraceCollector
from vizfold.backends.esmfold.schema import _read_fasta_and_hash
from vizfold.backends.esmfold.trace_adapter import (
    build_and_write_meta,
    write_structure,
    write_traces,
    write_trace_summary,
    write_attention_txt,
)

# HuggingFace ESMFold
try:
    from transformers import AutoTokenizer, EsmForProteinFolding
    HAS_ESM = True
except ImportError:
    HAS_ESM = False
    AutoTokenizer = None  # type: ignore
    EsmForProteinFolding = None  # type: ignore

# PDB conversion (bundled in transformers)
def _get_pdb_utils():
    try:
        from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
        from transformers.models.esm.openfold_utils.protein import Protein as OFProtein, to_pdb
        return atom14_to_atom37, OFProtein, to_pdb
    except ImportError:
        return None, None, None


LONG_SEQ_WARN_THRESHOLD = 400
HF_MODEL_DEFAULT = "facebook/esmfold_v1"

AA_1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "U": "SEC", "O": "PYL", "X": "UNK",
}


def _parse_layers_arg(layers_arg: Optional[str]) -> Optional[List[int]]:
    if not layers_arg or layers_arg.lower() == "all":
        return None
    indices = []
    for part in layers_arg.split(","):
        part = part.strip()
        if ":" in part:
            a, b = part.split(":", 1)
            indices.extend(range(int(a), int(b)))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def _parse_heads_arg(heads_arg: Optional[str]) -> Optional[List[int]]:
    if not heads_arg or heads_arg.lower() == "all":
        return None
    return [int(x.strip()) for x in heads_arg.split(",")]


def read_fasta(fasta_path: str) -> Tuple[str, str, str]:
    """Return (sequence, seq_id, fasta_hash)."""
    seq, fasta_hash = _read_fasta_and_hash(fasta_path)
    seq_id = "seq"
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                seq_id = line[1:].strip().split()[0]
                break
    return seq, seq_id, fasta_hash


class ESMFoldRunner(BackendBase):
    """
    Runs ESMFold inference and writes VizFold-compatible output.

    Uses HuggingFace EsmForProteinFolding (no fair-esm / OpenFold build).
    """

    def __init__(
        self,
        model_name: str = HF_MODEL_DEFAULT,
        device: str = "cpu",
        dtype: str = "float32",
        seed: Optional[int] = None,
        deterministic: bool = False,
    ):
        if not HAS_ESM:
            raise RuntimeError(
                "ESMFold backend requires transformers. Install with: pip install transformers torch"
            )
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.deterministic = deterministic
        self._model = None
        self._tokenizer = None

    # --- BackendBase interface ---

    def load_model(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Load the model. Returns the model object."""
        if model_name is not None:
            self.model_name = model_name
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self._load_model()

    def run_inference(
        self,
        fasta_path: str,
        out_dir: str,
        trace_cfg: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run inference and optionally write traces (BackendBase interface)."""
        trace_mode = "attention+activations"
        top_k = 50
        if trace_cfg:
            trace_mode = trace_cfg.get("trace_mode", trace_mode)
            top_k = trace_cfg.get("top_k", top_k)
        return self.run(
            fasta_path=fasta_path,
            out_dir=out_dir,
            trace_mode=trace_mode,
            top_k=top_k,
            **kwargs,
        )

    def supports_attention(self) -> bool:
        """Whether this backend can extract attention maps."""
        return True

    def supports_activations(self) -> bool:
        """Whether this backend can extract layer activations (hidden states)."""
        return True

    # --- Internal model loading ---

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self.seed is not None:
            torch.manual_seed(self.seed)
        if self.deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
            warnings.warn("Deterministic mode may reduce speed.", UserWarning)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = EsmForProteinFolding.from_pretrained(self.model_name, use_safetensors=True)
        self._model = self._model.eval()
        dtype_t = torch.float16 if self.dtype == "float16" else torch.float32
        self._model = self._model.to(device=self.device, dtype=dtype_t)
        return self._model

    def run(
        self,
        fasta_path: str,
        out_dir: str,
        trace_mode: str = "attention+activations",
        layers: Optional[str] = None,
        heads: Optional[str] = None,
        save_fp16: bool = False,
        top_k: int = 50,
        structure_traces: bool = False,
        log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run inference and write structure + optional traces.

        trace_mode: "attention" | "activations" | "attention+activations" | "none"
        layers: "all" or "0,1,2" or "0:12"
        heads: "all" or "0,1,2"
        structure_traces: if True, capture IPA attention weights and per-recycle
            backbone positions from the folding trunk structure module.
        """
        os.makedirs(out_dir, exist_ok=True)
        if log_path is None:
            log_path = os.path.join(out_dir, "logs.txt")

        def log(msg: str) -> None:
            with open(log_path, "a") as f:
                f.write(msg + "\n")
            print(msg)

        seq, seq_id, fasta_hash = read_fasta(fasta_path)
        seq_len = len(seq)

        if seq_len > LONG_SEQ_WARN_THRESHOLD and "attention" in trace_mode:
            log(
                f"Warning: sequence length {seq_len} > {LONG_SEQ_WARN_THRESHOLD}. "
                "Attention storage is N^2; consider --layers 0,1 or --trace_mode activations."
            )

        model = self._load_model()
        tokenizer = self._tokenizer
        want_attn = "attention" in trace_mode
        want_act = "activations" in trace_mode
        layer_list = _parse_layers_arg(layers)
        head_list = _parse_heads_arg(heads)

        collector = ESMFoldTraceCollector(
            want_attention=want_attn,
            want_activations=want_act,
            layer_indices=layer_list,
            head_indices=head_list,
            expected_seq_len=seq_len,
        )

        # Tokenize (HF ESMFold: add_special_tokens=False per standard usage)
        inputs = tokenizer(
            [seq],
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(device=self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device)

        # Hooks on model.esm capture ESM-2 encoder traces during forward.
        if trace_mode != "none":
            esm_trunk = getattr(model, "esm", model)
            collector.register_hooks(esm_trunk)
            # Hook the folding trunk to catch recycling iterations
            if hasattr(model, "trunk"):
                trunk_handle = model.trunk.register_forward_hook(collector._make_trunk_hook())
                collector._handles.append(trunk_handle)
            # Hook each evoformer block to capture per-block sequence/pair state
            collector.register_trunk_hooks(model)

        # Structure module hooks: IPA attention + per-recycle backbone
        sm_collector = None
        if structure_traces:
            sm_collector = StructureModuleTraceCollector()
            sm_collector.register_hooks(model)
            log("Structure module hooks registered (IPA attention + backbone).")

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        if trace_mode != "none":
            collector.remove_hooks()

            # Archive the recycled s_s and s_z tensors we caught
            if want_act and len(collector.recycled_s_s) > 0:
                num_iters = len(collector.recycled_s_s)
                log(f"[{self.model_name}] [{trace_mode}] Captured {num_iters} trunk recycling iterations.")
                for i in range(num_iters):
                    collector.activations[f"recycle_{i}_s_s"] = collector.recycled_s_s[i]
                    collector.activations[f"recycle_{i}_s_z"] = collector.recycled_s_z[i]

            # out.s_s: folding trunk single representations [B, N, 1024].
            # These are the per-residue embeddings produced by ESMFold's
            # structure module, complementing the ESM-2 encoder traces.
            if hasattr(out, 's_s') and out.s_s is not None:
                single_reps = out.s_s.squeeze(0).cpu()
                log(f"[{self.model_name}] [{trace_mode}] Extracted folding trunk s_s: {single_reps.shape}")
            else:
                log(f"[{self.model_name}] [{trace_mode}] out.s_s not found — folding trunk single representations missing.")

        if sm_collector is not None:
            sm_collector.remove_hooks()
            log(f"Structure module traces: {len(sm_collector.ipa_attention)} IPA blocks, "
                f"{len(sm_collector.backbone_positions)} recycle iterations.")

        log(f"Forward pass complete. Captured {len(collector.attention)} attention layers, "
            f"{len(collector.activations)} activation layers.")

        # Structure: PDB + optional coords tensor
        pdb_str, coords = self._extract_structure(out, seq, log)
        struct_paths = write_structure(out_dir, pdb_str, coords)
        log(f"Structure written: {struct_paths}")

        # Traces
        shapes_recorded = {"attention": {}, "activations": {}}
        if trace_mode != "none" and (collector.attention or collector.activations):
            attn_idx, act_idx = write_traces(
                out_dir,
                collector,
                save_fp16=save_fp16,
                layer_indices=layer_list,
                head_indices=head_list,
            )
            for k, v in attn_idx.items():
                shapes_recorded["attention"][k] = v.get("shape", [])
            for k, v in act_idx.items():
                shapes_recorded["activations"][k] = v.get("shape", [])
            # Save per-block evoformer intermediates + final s_s/s_z into trace/trunk/
            trunk_tensors = dict(collector.trunk_blocks)  # block_000_seq, block_000_pair, ...
            # Add final trunk outputs: recycled_s_s[-1] == out.s_s (last recycle iteration).
            # Saved here rather than from out.s_s to avoid a redundant copy.
            if collector.recycled_s_s:
                trunk_tensors["s_s"] = collector.recycled_s_s[-1]  # [L, C_s]
            if collector.recycled_s_z:
                trunk_tensors["s_z"] = collector.recycled_s_z[-1]  # [L, L, C_z]

            if trunk_tensors:
                trunk_dir = os.path.join(out_dir, "trace", "trunk")
                os.makedirs(trunk_dir, exist_ok=True)
                shapes_recorded["trunk"] = {}
                for key, t in trunk_tensors.items():
                    path = os.path.join(trunk_dir, f"{key}.pt")
                    torch.save(t if not save_fp16 else t.half(), path)
                    shapes_recorded["trunk"][key] = {
                        "path": os.path.relpath(path, out_dir),
                        "shape": list(t.shape),
                    }
                log(f"Evoformer trunk intermediates written to {trunk_dir} "
                    f"({len(trunk_tensors)} tensors: "
                    f"{len(collector.trunk_blocks)} blocks + "
                    f"{len(trunk_tensors) - len(collector.trunk_blocks)} final)")

            try:
                write_trace_summary(out_dir, collector)
            except Exception as e:
                log(f"Warning: trace summary failed: {e}")

            # VizFold-compatible text-file attention export
            if want_attn and collector.attention:
                txt_dir = write_attention_txt(out_dir, collector, top_k=top_k)
                if txt_dir:
                    attn_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
                    shapes_recorded["attention_files"] = attn_files
                    log(f"VizFold text attention saved to {txt_dir} ({len(attn_files)} files)")

        # Write structure module traces (IPA attention + backbone per recycle)
        if sm_collector is not None:
            sm_dir = os.path.join(out_dir, "trace", "structure_module")
            ipa_dir = os.path.join(sm_dir, "ipa_attention")
            bb_dir = os.path.join(sm_dir, "backbone")
            os.makedirs(ipa_dir, exist_ok=True)
            os.makedirs(bb_dir, exist_ok=True)

            shapes_recorded["structure_module"] = {"ipa_attention": {}, "backbone": {}}

            for key, t in sm_collector.ipa_attention.items():
                path = os.path.join(ipa_dir, f"{key}.pt")
                torch.save(t.cpu() if not save_fp16 else t.cpu().half(), path)
                shapes_recorded["structure_module"]["ipa_attention"][key] = {
                    "path": os.path.relpath(path, out_dir),
                    "shape": list(t.shape),
                }

            for key, t in sm_collector.backbone_positions.items():
                path = os.path.join(bb_dir, f"{key}_positions.pt")
                torch.save(t if not save_fp16 else t.half(), path)
                shapes_recorded["structure_module"]["backbone"][f"{key}_positions"] = {
                    "path": os.path.relpath(path, out_dir),
                    "shape": list(t.shape),
                }

            for key, t in sm_collector.sm_states.items():
                path = os.path.join(bb_dir, f"{key}_states.pt")
                torch.save(t if not save_fp16 else t.half(), path)
                shapes_recorded["structure_module"]["backbone"][f"{key}_states"] = {
                    "path": os.path.relpath(path, out_dir),
                    "shape": list(t.shape),
                }

            log(f"Structure module traces written: "
                f"{len(sm_collector.ipa_attention)} IPA attention maps, "
                f"{len(sm_collector.backbone_positions)} backbone snapshots.")

        layer_count = len(collector.attention) if trace_mode != "none" else 0
        head_count = 0
        if collector.attention:
            first_attn = next(iter(collector.attention.values()))
            if first_attn.dim() == 4:
                head_count = first_attn.shape[1]
            elif first_attn.dim() == 3:
                head_count = first_attn.shape[0]

        build_and_write_meta(
            out_dir=out_dir,
            model_name=self.model_name,
            fasta_path=os.path.abspath(fasta_path),
            device=self.device,
            dtype=self.dtype,
            seq_len=seq_len,
            fasta_hash=fasta_hash,
            layer_count=layer_count,
            head_count=head_count,
            trace_mode=trace_mode,
            shapes_recorded=shapes_recorded,
            seed=self.seed,
            deterministic=self.deterministic,
            save_fp16=save_fp16,
            top_k=top_k,
        )
        log("meta.json written.")

        return {
            "structure": struct_paths,
            "out_dir": out_dir,
            "trace_mode": trace_mode,
            "attention_layers": len(collector.attention),
            "activation_layers": len(collector.activations),
        }

    def _extract_structure(self, out: Any, seq: str, log) -> Tuple[Optional[str], Optional[torch.Tensor]]:
        """Extract PDB string and coordinates from model output."""
        pdb_str = None
        coords = None
        atom14_to_atom37, OFProtein, to_pdb = _get_pdb_utils()

        def _get(key: str):
            if hasattr(out, key):
                return getattr(out, key)
            if isinstance(out, dict) and key in out:
                return out[key]
            return None

        positions = _get("positions")
        if positions is None:
            log("Warning: no positions in model output; structure/ may be incomplete.")
            return pdb_str, coords

        # Fallback chain: full OpenFold PDB conversion → CA-only minimal PDB.
        # When transformers bundles openfold_utils we get proper all-atom PDB;
        # otherwise we write a CA-only PDB so downstream tools still have geometry.
        try:
            if atom14_to_atom37 and OFProtein is not None and to_pdb is not None:
                # ESMFold returns positions as [recycling_iters, B, N, 14, 3];
                # take the last iteration for the final refined structure.
                pos = positions[-1] if positions.dim() == 5 else positions
                final_atom37 = atom14_to_atom37(pos, out)
                coords = final_atom37
                out_cpu = {}
                for k in ("aatype", "atom37_atom_exists", "residue_index", "plddt", "chain_index"):
                    v = _get(k)
                    if v is not None:
                        out_cpu[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                pos_np = final_atom37.cpu().numpy()
                pred = OFProtein(
                    aatype=out_cpu["aatype"][0],
                    atom_positions=pos_np[0],
                    atom_mask=out_cpu["atom37_atom_exists"][0],
                    residue_index=out_cpu["residue_index"][0] + 1,
                    b_factors=out_cpu["plddt"][0],
                    chain_index=out_cpu.get("chain_index", [None])[0] if "chain_index" in out_cpu else None,
                )
                pdb_str = to_pdb(pred)
            else:
                pos = positions
                if pos.dim() == 5:
                    pos = pos[-1, 0]
                elif pos.dim() == 4:
                    pos = pos[0]
                coords = pos
                pdb_str = _coords_to_minimal_pdb(coords, seq)
        except Exception as e:
            log(f"Warning: PDB conversion failed ({e}); writing minimal PDB.")
            try:
                if positions is not None:
                    pos = positions
                    if pos.dim() == 5:
                        pos = pos[-1, 0]
                    elif pos.dim() == 4:
                        pos = pos[0]
                    pdb_str = _coords_to_minimal_pdb(pos, seq)
                    coords = pos
            except Exception:
                pass

        if pdb_str is None and coords is not None:
            pdb_str = _coords_to_minimal_pdb(coords, seq)
        if pdb_str is None:
            log("Warning: no PDB output from model; structure/ may be incomplete.")

        return pdb_str, coords


def _coords_to_minimal_pdb(coords: torch.Tensor, seq: str) -> str:
    """Write minimal CA-only PDB from coords [N, 14, 3] or [N, 37, 3] or [N, 3]."""
    if coords.dim() == 3:
        ca = coords[:, 1, :]  # atom37/atom14 order: N=0, CA=1, C=2, ...
    else:
        ca = coords
    lines = []
    for i in range(min(ca.shape[0], len(seq))):
        a = ca[i].float().cpu().numpy()
        res3 = AA_1TO3.get(seq[i], "UNK")
        # PDB ATOM format: columns 1-6 record type, 7-11 serial, 13-16 name,
        # 17 altLoc, 18-20 resName, 22 chainID, 23-26 resSeq, 27 iCode,
        # 31-38 x, 39-46 y, 47-54 z, 55-60 occupancy, 61-66 tempFactor
        lines.append(
            f"ATOM  {i+1:5d}  CA  {res3:>3s} A{i+1:4d}    "
            f"{a[0]:8.3f}{a[1]:8.3f}{a[2]:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"
