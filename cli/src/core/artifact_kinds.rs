//! What a folding run produces, as kinds rather than as files.
//!
//! Two things live here and must stay together: the catalog seeded into `artifact_types`, and the
//! rule that decides which kind a produced file is. The catalog carries the classification and the
//! hints for showing it; the artifact rows carry the instances. A file is classified from its path
//! alone, so registering a run is a walk plus a lookup -- no backend has to declare what it wrote.

use std::path::Path;

/// A row of the `artifact_types` catalog.
pub struct ArtifactKind {
    pub slug: &'static str,
    pub label: &'static str,
    /// What this kind is usually written as. The instance records what it actually is.
    pub default_format: &'static str,
    /// How the dashboard presents it: `embedded` in a viewer, `download` as a link, `internal`
    /// for something it reads but never shows.
    pub display_mode: &'static str,
    /// Which viewer the dashboard reaches for when the instance's format allows it.
    pub viewer_kind: &'static str,
    pub description: &'static str,
    /// What an instance's `metadata_json` carries, filled in at registration from the file name.
    pub metadata_schema_json: &'static str,
}

const NO_METADATA: &str = "{}";

/// Every kind a fold actually produces. Nothing speculative: each row below is written by
/// `scripts/openfold/run_pretrained_openfold.py`, OpenFold's evoformer save sites, or
/// `backends/esmfold/esmfold/{inference,trace_adapter}.py`.
pub const KINDS: &[ArtifactKind] = &[
    ArtifactKind {
        slug: "protein_structure",
        label: "Protein structure",
        default_format: "pdb",
        display_mode: "embedded",
        viewer_kind: "structure_viewer",
        description: "The predicted structure. OpenFold writes relaxed and unrelaxed forms per \
                      target; ESMFold writes one, plus its coordinates as a tensor.",
        metadata_schema_json: r#"{"type":"object","properties":{"target":{"type":"string"},"relaxed":{"type":"boolean"}}}"#,
    },
    ArtifactKind {
        slug: "attention_map",
        label: "Attention map",
        default_format: "txt",
        display_mode: "embedded",
        viewer_kind: "arc_diagram",
        description: "Top-k attention edges for one layer, per head, as `<residue> <residue> \
                      <weight>` lines. What the arc diagrams are drawn from.",
        metadata_schema_json: r#"{"type":"object","properties":{"target":{"type":"string"},"attention":{"type":"string","enum":["msa_row","triangle_start"]},"layer":{"type":"integer"},"residue":{"type":"string"},"key":{"type":"string"}}}"#,
    },
    ArtifactKind {
        slug: "attention_tensor",
        label: "Attention tensor",
        default_format: "pt",
        display_mode: "download",
        viewer_kind: "download_link",
        description: "A dense attention array: OpenFold's `.npz` per layer, ESMFold's per-layer \
                      and IPA tensors.",
        metadata_schema_json: r#"{"type":"object","properties":{"target":{"type":"string"},"attention":{"type":"string"},"layer":{"type":"integer"},"residue":{"type":"string"},"key":{"type":"string"}}}"#,
    },
    ArtifactKind {
        slug: "activation_tensor",
        label: "Activation tensor",
        default_format: "pt",
        display_mode: "download",
        viewer_kind: "download_link",
        description: "What the model computed on the way: per-layer activations, trunk block and \
                      recycle states, and the structure module's backbone positions.",
        metadata_schema_json: r#"{"type":"object","properties":{"target":{"type":"string"},"key":{"type":"string"}}}"#,
    },
    ArtifactKind {
        slug: "model_output_archive",
        label: "Model output archive",
        default_format: "pkl",
        display_mode: "download",
        viewer_kind: "download_link",
        description: "OpenFold's whole output dictionary, pickled, written when a run asks to \
                      save outputs.",
        metadata_schema_json: r#"{"type":"object","properties":{"target":{"type":"string"}}}"#,
    },
    ArtifactKind {
        slug: "trace_summary",
        label: "Trace summary",
        default_format: "json",
        display_mode: "embedded",
        viewer_kind: "stat_panel",
        description: "Per-layer attention and activation statistics, which the dashboard plots \
                      rather than making the browser read every tensor.",
        metadata_schema_json: NO_METADATA,
    },
    ArtifactKind {
        slug: "trace_index",
        label: "Trace index",
        default_format: "json",
        display_mode: "internal",
        viewer_kind: "viewer_registry",
        description: "Which tensor is where, with its dtype and shape. Read to list the trace, \
                      never shown on its own.",
        metadata_schema_json: NO_METADATA,
    },
    ArtifactKind {
        slug: "run_metadata",
        label: "Run metadata",
        default_format: "json",
        display_mode: "embedded",
        viewer_kind: "key_values",
        description: "What the run was: model, device, dtype, sequence length, layer and head \
                      counts, and the timings it recorded.",
        metadata_schema_json: r#"{"type":"object","properties":{"stage":{"type":"string","enum":["inference","relaxation"]}}}"#,
    },
    ArtifactKind {
        slug: "sequence_alignment",
        label: "Sequence alignment",
        default_format: "a3m",
        display_mode: "download",
        viewer_kind: "download_link",
        description: "The MSA a run searched for itself, written under `alignments/<target>/` when \
                      it was not handed precomputed ones.",
        metadata_schema_json: r#"{"type":"object","properties":{"target":{"type":"string"},"database":{"type":"string"}}}"#,
    },
    ArtifactKind {
        slug: "template_hits",
        label: "Template hits",
        default_format: "hhr",
        display_mode: "download",
        viewer_kind: "download_link",
        description: "The structural template search beside that MSA.",
        metadata_schema_json: r#"{"type":"object","properties":{"target":{"type":"string"}}}"#,
    },
    ArtifactKind {
        slug: "run_log",
        label: "Run log",
        default_format: "txt",
        display_mode: "download",
        viewer_kind: "download_link",
        description: "What the backend printed while it ran.",
        metadata_schema_json: NO_METADATA,
    },
    ArtifactKind {
        slug: "run_output_directory",
        label: "Run output directory",
        default_format: "directory",
        display_mode: "internal",
        viewer_kind: "directory_link",
        description: "The workspace itself, which is how anything reading a run finds the rest.",
        metadata_schema_json: NO_METADATA,
    },
    ArtifactKind {
        slug: "run_file",
        label: "Run file",
        default_format: "",
        display_mode: "download",
        viewer_kind: "download_link",
        description: "Something the run wrote that no more specific kind claims. Every produced \
                      file is registered, so this is what keeps that true.",
        metadata_schema_json: NO_METADATA,
    },
];

/// Which kind a produced file is, from its path relative to the run's output directory.
///
/// Ordering is the whole of the rule: the most specific claim wins, and `run_file` closes the set
/// so that every file a run wrote lands somewhere.
pub fn classify(relative_path: &Path) -> &'static str {
    let path = relative_path.to_string_lossy().replace('\\', "/");
    let name = relative_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();

    // The attention dumps, which are the same names wherever a backend puts them: the top-k text
    // is read by the arc diagrams, the dense array beside it is a download.
    if let Some(captures) = attention_captures(&name) {
        return if captures.3 == "txt" {
            "attention_map"
        } else {
            "attention_tensor"
        };
    }

    // ESMFold's trace tree, which says what a tensor is by where it sits.
    if path.starts_with("trace/attention/")
        || path.starts_with("trace/structure_module/ipa_attention/")
    {
        return "attention_tensor";
    }
    if path.starts_with("trace/activations/")
        || path.starts_with("trace/trunk/")
        || path.starts_with("trace/structure_module/backbone/")
    {
        return "activation_tensor";
    }
    if path == "trace/summary.json" {
        return "trace_summary";
    }
    if path == "trace/index.json" {
        return "trace_index";
    }

    if name == "meta.json" || name == "timings.json" {
        return "run_metadata";
    }
    if name == "logs.txt" {
        return "run_log";
    }
    // A run without precomputed alignments searches for its own, into the workspace.
    if path.starts_with("alignments/") {
        return match relative_path
            .extension()
            .map(|extension| extension.to_string_lossy().to_lowercase())
            .as_deref()
        {
            Some("hhr") => "template_hits",
            Some("a3m" | "sto" | "fasta" | "fa") => "sequence_alignment",
            _ => "run_file",
        };
    }
    if name.ends_with("_output_dict.pkl") {
        return "model_output_archive";
    }

    // A structure by extension, wherever it was written -- and ESMFold's coordinate tensor, which
    // is the same thing in a form no viewer can embed.
    if let Some("pdb" | "cif" | "ent") = relative_path
        .extension()
        .map(|extension| extension.to_string_lossy().to_lowercase())
        .as_deref()
    {
        return "protein_structure";
    }
    if path == "structure/predicted.pt" {
        return "protein_structure";
    }

    "run_file"
}

/// What an instance of that kind carries: the coordinates its name encodes, so nothing downstream
/// has to parse the file name again.
pub fn metadata(relative_path: &Path, slug: &str, tags: &[String]) -> serde_json::Value {
    let name = relative_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut metadata = serde_json::Map::new();

    if let Some(target) = target_of(relative_path, tags) {
        metadata.insert("target".into(), target.into());
    }

    match slug {
        "attention_map" | "attention_tensor" => {
            if let Some((kind, layer, residue, _)) = attention_captures(&name) {
                metadata.insert("attention".into(), kind.into());
                metadata.insert("layer".into(), layer.into());
                if let Some(residue) = residue {
                    metadata.insert("residue".into(), residue.into());
                }
            } else if let Some(key) = stem(&name) {
                metadata.insert("key".into(), key.into());
            }
        }
        "activation_tensor" => {
            if let Some(key) = stem(&name) {
                metadata.insert("key".into(), key.into());
            }
        }
        "protein_structure" => {
            if name.contains("_relaxed.") || name.contains("_unrelaxed.") {
                metadata.insert("relaxed".into(), (!name.contains("_unrelaxed.")).into());
            }
        }
        // Two files share this name at different depths: inference timings at the workspace root,
        // relaxation timings beside the structures.
        "run_metadata" if name == "timings.json" => {
            let stage = if relative_path
                .parent()
                .is_none_or(|parent| parent.as_os_str().is_empty())
            {
                "inference"
            } else {
                "relaxation"
            };
            metadata.insert("stage".into(), stage.into());
        }
        _ => {}
    }

    serde_json::Value::Object(metadata)
}

/// The target a file belongs to, where the layout names one: OpenFold nests attention under
/// `attention/<tag>/` and prefixes its structures with the tag. ESMFold folds one target and
/// names nothing, so a lone run's files carry no target.
///
/// `tags` is what the run says it folded. Matching against those rather than against a separator
/// is the only reliable read: a file is `<tag>_<config_preset>_relaxed.pdb`, and OpenFold's presets
/// include `seq_model_esm1b_ptm` and `finetuning_ptm` -- one of which contains the separator a
/// naive split would cut at, and the other of which does not contain it at all.
fn target_of(relative_path: &Path, tags: &[String]) -> Option<String> {
    let path = relative_path.to_string_lossy().replace('\\', "/");
    if let Some(rest) = path.strip_prefix("attention/")
        && let Some((head, tail)) = rest.split_once('/')
        && !tail.is_empty()
    {
        return Some(head.to_owned());
    }

    let name = relative_path.file_name()?.to_string_lossy().into_owned();
    // Longest first, so `1UBQ_1` never shadows `1UBQ_10`.
    let mut candidates: Vec<&String> = tags.iter().collect();
    candidates.sort_by_key(|tag| std::cmp::Reverse(tag.len()));
    candidates
        .into_iter()
        .find(|tag| name.starts_with(&format!("{tag}_")))
        .map(|tag| tag.to_owned())
}

fn stem(name: &str) -> Option<String> {
    Path::new(name)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
}

/// `(attention kind, layer, query residue, extension)` for the names the backends write:
/// `(msa_row|triangle_start)_attn_layer<L>[_residue_idx_<R|avg>].(txt|npz|pt)`, which
/// `save_attention_topk` in OpenFold's evoformer owns and ESMFold's adapter reuses.
fn attention_captures(name: &str) -> Option<(&'static str, i64, Option<String>, String)> {
    let rest = name
        .strip_suffix(".txt")
        .map(|rest| (rest, "txt"))
        .or_else(|| {
            name.strip_suffix(".npz")
                .map(|rest| (rest, "npz"))
                .or_else(|| name.strip_suffix(".pt").map(|rest| (rest, "pt")))
        })?;
    let (body, extension) = rest;

    let (kind, tail) = body
        .strip_prefix("msa_row_attn_layer")
        .map(|tail| ("msa_row", tail))
        .or_else(|| {
            body.strip_prefix("triangle_start_attn_layer")
                .map(|tail| ("triangle_start", tail))
        })?;

    let (layer, residue) = match tail.split_once("_residue_idx_") {
        Some((layer, residue)) => (layer, Some(residue.to_owned())),
        None => (tail, None),
    };
    let layer = layer.parse::<i64>().ok()?;
    Some((kind, layer, residue, extension.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every file the two backends write, taken from the code that writes it. A kind that stops
    /// matching its real file is the failure this guards.
    const OPENFOLD: &[(&str, &str)] = &[
        // `make_output_directory` nests the structures under predictions/, so nothing here may
        // depend on a file sitting at the workspace root.
        (
            "predictions/1UBQ_1_model_1_ptm_relaxed.pdb",
            "protein_structure",
        ),
        ("predictions/timings.json", "run_metadata"),
        ("alignments/1UBQ_1/uniref90_hits.sto", "sequence_alignment"),
        (
            "alignments/1UBQ_1/bfd_uniclust_hits.a3m",
            "sequence_alignment",
        ),
        ("alignments/1UBQ_1/hhsearch_output.hhr", "template_hits"),
        ("1UBQ_1_model_1_ptm_relaxed.pdb", "protein_structure"),
        ("1UBQ_1_model_1_ptm_unrelaxed.pdb", "protein_structure"),
        ("6KWC_1_model_1_ptm_unrelaxed.cif", "protein_structure"),
        ("1UBQ_1_model_1_ptm_output_dict.pkl", "model_output_archive"),
        ("timings.json", "run_metadata"),
        ("attention/1UBQ_1/msa_row_attn_layer47.txt", "attention_map"),
        (
            "attention/1UBQ_1/triangle_start_attn_layer47_residue_idx_18.txt",
            "attention_map",
        ),
        (
            "attention/1UBQ_1/triangle_start_attn_layer47_residue_idx_avg.txt",
            "attention_map",
        ),
        (
            "attention/1UBQ_1/msa_row_attn_layer47.npz",
            "attention_tensor",
        ),
    ];

    const ESMFOLD: &[(&str, &str)] = &[
        ("meta.json", "run_metadata"),
        ("logs.txt", "run_log"),
        ("trace/activations/recycle_0_s_s.pt", "activation_tensor"),
        (
            "trace/structure_module/backbone/recycle_00_states.pt",
            "activation_tensor",
        ),
        ("structure/predicted.pdb", "protein_structure"),
        ("structure/predicted.pt", "protein_structure"),
        ("trace/index.json", "trace_index"),
        ("trace/summary.json", "trace_summary"),
        ("trace/attention/layer_000.pt", "attention_tensor"),
        ("trace/activations/layer_000.pt", "activation_tensor"),
        ("trace/trunk/block_000_seq.pt", "activation_tensor"),
        ("trace/trunk/s_z.pt", "activation_tensor"),
        (
            "trace/structure_module/ipa_attention/recycle_0_block_1.pt",
            "attention_tensor",
        ),
        (
            "trace/structure_module/backbone/recycle_0_positions.pt",
            "activation_tensor",
        ),
        ("attention/msa_row_attn_layer0.txt", "attention_map"),
    ];

    #[test]
    fn every_real_output_lands_on_its_kind() {
        for (path, want) in OPENFOLD.iter().chain(ESMFOLD) {
            assert_eq!(classify(Path::new(path)), *want, "{path}");
        }
    }

    #[test]
    fn anything_else_is_still_registered() {
        // The catch-all is what makes "every produced file is typed" true rather than aspirational.
        for path in [
            "notes.txt",
            // A staging FASTA a crashed run left behind, and a cached weights conversion: real
            // files, but not results anyone folded for.
            "tmp_1234.fasta",
            "converted_checkpoint.bin",
            "weird",
        ] {
            assert_eq!(classify(Path::new(path)), "run_file", "{path}");
        }
    }

    #[test]
    fn every_kind_the_classifier_names_is_in_the_catalog() {
        let slugs: Vec<&str> = KINDS.iter().map(|kind| kind.slug).collect();
        for (path, _) in OPENFOLD.iter().chain(ESMFOLD) {
            let slug = classify(Path::new(path));
            assert!(slugs.contains(&slug), "{slug} is not seeded");
        }
        assert!(slugs.contains(&"run_file"));
        assert!(slugs.contains(&"run_output_directory"));
    }

    #[test]
    fn an_instance_carries_the_coordinates_its_name_encodes() {
        let tags = [String::from("1UBQ_1"), String::from("6KWC_1")];
        let attention = metadata(
            Path::new("attention/1UBQ_1/triangle_start_attn_layer47_residue_idx_18.txt"),
            "attention_map",
            &tags,
        );
        assert_eq!(attention["target"], "1UBQ_1");
        assert_eq!(attention["attention"], "triangle_start");
        assert_eq!(attention["layer"], 47);
        assert_eq!(attention["residue"], "18");

        let relaxed = metadata(
            Path::new("predictions/1UBQ_1_model_1_ptm_relaxed.pdb"),
            "protein_structure",
            &tags,
        );
        assert_eq!(relaxed["target"], "1UBQ_1");
        assert_eq!(relaxed["relaxed"], true);
        assert_eq!(
            metadata(
                Path::new("predictions/1UBQ_1_model_1_ptm_unrelaxed.pdb"),
                "protein_structure",
                &tags
            )["relaxed"],
            false
        );

        // ESMFold folds one target and names none, so nothing is invented for it -- including
        // the relaxed/unrelaxed claim, which only OpenFold's names make.
        let lone = metadata(
            Path::new("structure/predicted.pdb"),
            "protein_structure",
            &tags,
        );
        assert_eq!(lone.get("target"), None);
        assert_eq!(lone.get("relaxed"), None);

        // OpenFold's own preset names defeat a separator-based read: one contains the separator,
        // the other has none. Matching the run's tags is what survives both.
        for name in [
            "predictions/1UBQ_1_seq_model_esm1b_ptm_relaxed.pdb",
            "predictions/1UBQ_1_finetuning_ptm_relaxed.pdb",
            "predictions/1UBQ_1_model_1_ptm_relaxed.pdb",
        ] {
            assert_eq!(
                metadata(Path::new(name), "protein_structure", &tags)["target"],
                "1UBQ_1",
                "{name}"
            );
        }
        // A longer tag wins, so 1UBQ_1 never claims 1UBQ_10's files.
        let both = [String::from("1UBQ_1"), String::from("1UBQ_10")];
        assert_eq!(
            metadata(
                Path::new("predictions/1UBQ_10_model_1_ptm_relaxed.pdb"),
                "protein_structure",
                &both
            )["target"],
            "1UBQ_10"
        );

        // The same basename at two depths times two different things.
        assert_eq!(
            metadata(Path::new("timings.json"), "run_metadata", &tags)["stage"],
            "inference"
        );
        assert_eq!(
            metadata(Path::new("predictions/timings.json"), "run_metadata", &tags)["stage"],
            "relaxation"
        );
    }

    /// The schema is a claim about the instance. A key `metadata()` writes that the schema does
    /// not name is exactly the drift this file exists to prevent.
    #[test]
    fn what_metadata_writes_is_what_the_schema_declares() {
        let tags = [String::from("1UBQ_1")];
        let cases = [
            ("attention/1UBQ_1/msa_row_attn_layer4.txt", "attention_map"),
            (
                "attention/1UBQ_1/msa_row_attn_layer4.npz",
                "attention_tensor",
            ),
            ("trace/activations/layer_000.pt", "activation_tensor"),
            (
                "predictions/1UBQ_1_model_1_ptm_relaxed.pdb",
                "protein_structure",
            ),
            ("timings.json", "run_metadata"),
            (
                "predictions/1UBQ_1_model_1_ptm_output_dict.pkl",
                "model_output_archive",
            ),
            ("alignments/1UBQ_1/uniref90_hits.sto", "sequence_alignment"),
        ];
        for (path, slug) in cases {
            let kind = KINDS.iter().find(|kind| kind.slug == slug).expect(slug);
            let schema: serde_json::Value =
                serde_json::from_str(kind.metadata_schema_json).expect(slug);
            let declared = schema["properties"].as_object();
            let written = metadata(Path::new(path), slug, &tags);
            for key in written.as_object().expect("an object").keys() {
                assert!(
                    declared.is_some_and(|properties| properties.contains_key(key)),
                    "{slug} writes `{key}`, which its metadata schema does not declare"
                );
            }
        }
    }

    #[test]
    fn the_catalog_vocabulary_is_closed() {
        for kind in KINDS {
            assert!(
                matches!(kind.display_mode, "embedded" | "download" | "internal"),
                "{} has an unknown display mode {}",
                kind.slug,
                kind.display_mode
            );
            assert!(
                serde_json::from_str::<serde_json::Value>(kind.metadata_schema_json).is_ok(),
                "{} has an unparseable metadata schema",
                kind.slug
            );
        }
    }

    #[test]
    fn the_parser_accepts_the_names_the_backends_write_and_nothing_else() {
        assert!(attention_captures("msa_row_attn_layer47.txt").is_some());
        assert!(attention_captures("triangle_start_attn_layer47_residue_idx_avg.npz").is_some());
        assert!(attention_captures("msa_row_attn_layerX.txt").is_none());
        assert!(attention_captures("msa_row_attn_layer47.png").is_none());
    }
}
