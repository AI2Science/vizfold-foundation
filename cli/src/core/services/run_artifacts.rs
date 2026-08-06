use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
};

use sea_orm::{DatabaseConnection, DbErr, EntityTrait};

use crate::core::{
    artifact_kinds,
    entities::{artifact_types, artifacts, runs},
    output_locations::resolve_output_location,
};

use super::artifacts::{self as artifact_service, RecordArtifactInput};

/// Everything a run wrote, as `(path relative to the workspace, kind)`. The classification is
/// `artifact_kinds::classify`, so the kinds registered against a run and the kinds the catalog
/// holds are one set rather than two that must be kept in step.
pub fn produced_files(workspace: &Path) -> Vec<(PathBuf, &'static str)> {
    let mut found = Vec::new();
    collect(workspace, workspace, &mut found);
    found.sort_by(|left, right| left.0.cmp(&right.0));
    found
}

fn collect(root: &Path, dir: &Path, found: &mut Vec<(PathBuf, &'static str)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            collect(root, &path, found);
        } else if file_type.is_file()
            && let Ok(relative) = path.strip_prefix(root)
        {
            found.push((relative.to_path_buf(), artifact_kinds::classify(relative)));
        }
    }
}

/// How many instances of each kind a pass registered.
pub type RegistrationReport = BTreeMap<&'static str, usize>;

/// The pass plus the run's whole artifact list. Production wants the report, not the list, so this
/// is the read tests and any future caller that needs both use.
#[cfg(test)]
pub async fn register_known_run_artifacts(
    db: &DatabaseConnection,
    run_id: i32,
) -> Result<Vec<artifacts::Model>, DbErr> {
    register_run_artifacts(db, run_id).await?;
    artifact_service::list_artifacts_for_run(db, run_id).await
}

/// The pass itself, reporting what it added this time.
pub async fn register_run_artifacts(
    db: &DatabaseConnection,
    run_id: i32,
) -> Result<RegistrationReport, DbErr> {
    let run = runs::Entity::find_by_id(run_id)
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom(format!("run {run_id} does not exist")))?;
    let profile = super::require_profile(db, run.invocation_profile_id).await?;
    let workspace = resolve_output_location(&profile, &run)?;

    let mut report = RegistrationReport::new();
    if !workspace.is_dir() {
        return Ok(report);
    }

    let types = artifact_type_ids(db).await?;
    // One query for what is already recorded: re-registering a large trace would otherwise be a
    // round trip per file.
    let registered: HashMap<String, artifacts::Model> =
        artifact_service::list_artifacts_for_run(db, run_id)
            .await?
            .into_iter()
            .map(|artifact| (artifact.storage_uri.clone(), artifact))
            .collect();

    // What the run says it folded: a produced file names its target with one of these, and a
    // separator-based guess cannot survive OpenFold's preset names.
    let tags: Vec<String> = run.input_id.split('+').map(str::to_owned).collect();

    // The workspace first: everything that reads a run finds the rest through it.
    let mut recording = vec![(
        workspace.clone(),
        "run_output_directory",
        "directory".to_owned(),
        serde_json::json!({}),
    )];
    for (relative, slug) in produced_files(&workspace) {
        let format = relative
            .extension()
            .map(|extension| extension.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let metadata = artifact_kinds::metadata(&relative, slug, &tags);
        recording.push((workspace.join(&relative), slug, format, metadata));
    }

    for (path, slug, format, metadata) in recording {
        let storage_uri = path.display().to_string();
        let artifact_type_id = *types
            .get(slug)
            .ok_or_else(|| DbErr::Custom(format!("artifact type '{slug}' is missing")))?;
        if let Some(existing) = registered.get(&storage_uri) {
            // Already recorded. Re-registering after the classifier learned a new kind should
            // move the row onto it rather than leave the file typed by an older rule forever.
            if existing.artifact_type_id != artifact_type_id {
                artifact_service::reclassify_artifact(
                    db,
                    existing.id,
                    artifact_type_id,
                    metadata.to_string(),
                )
                .await?;
                *report.entry(slug).or_default() += 1;
            }
            continue;
        }
        artifact_service::record_artifact_manifest_entry(
            db,
            RecordArtifactInput {
                run_id,
                artifact_type_id,
                format,
                storage_uri,
                metadata_json: metadata.to_string(),
            },
        )
        .await?;
        *report.entry(slug).or_default() += 1;
    }

    Ok(report)
}

async fn artifact_type_ids(db: &DatabaseConnection) -> Result<HashMap<String, i32>, DbErr> {
    Ok(artifact_types::Entity::find()
        .all(db)
        .await?
        .into_iter()
        .map(|artifact_type| (artifact_type.slug, artifact_type.id))
        .collect())
}

/// The kinds a run holds, by slug: the shape of a run without listing its every file.
pub async fn artifact_counts_by_kind(
    db: &DatabaseConnection,
    run_id: i32,
) -> Result<BTreeMap<String, usize>, DbErr> {
    let types: HashMap<i32, String> = artifact_types::Entity::find()
        .all(db)
        .await?
        .into_iter()
        .map(|artifact_type| (artifact_type.id, artifact_type.slug))
        .collect();
    let mut counts = BTreeMap::new();
    for artifact in artifact_service::list_artifacts_for_run(db, run_id).await? {
        let slug = types
            .get(&artifact.artifact_type_id)
            .cloned()
            .unwrap_or_else(|| artifact.artifact_type_id.to_string());
        *counts.entry(slug).or_default() += 1;
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use sea_orm::{ColumnTrait, DbErr, EntityTrait, QueryFilter};
    use serde_json::json;

    use crate::core::{
        entities::runs as run_entity,
        services::{
            model_invocation_profiles::{self, RegisterModelInvocationProfileInput},
            runs::{self, SubmitRunInput},
        },
    };

    use super::{produced_files, register_known_run_artifacts, register_run_artifacts};

    async fn run_with_output_root(
        db: &sea_orm::DatabaseConnection,
        output_root: &PathBuf,
    ) -> Result<run_entity::Model, DbErr> {
        let (backend, target) = crate::core::test_support::local_backend_and_target(
            db,
            "openfold-artifact-test",
            "local-artifact-test",
        )
        .await?;
        let profile = model_invocation_profiles::register_model_invocation_profile(
            db,
            RegisterModelInvocationProfileInput {
                model_backend_id: backend.id,
                execution_target_id: target.id,
                invocation_kind: "local_subprocess".into(),
                config_json: json!({ "output_location": output_root }).to_string(),
            },
        )
        .await?;
        runs::submit_run(
            db,
            SubmitRunInput {
                model_backend_id: backend.id,
                execution_target_id: target.id,
                invocation_profile_id: profile.id,
                status: "completed".into(),
                input_id: "1UBQ_1".into(),
                input_sequence: "MSTNPKPQRITF".into(),
                model_parameters_json: "{}".into(),
                execution_parameters_json: "{}".into(),
                provenance_json: None,
            },
        )
        .await
    }

    /// One of everything the two backends write.
    fn write_run_outputs(workspace: &Path) {
        for dir in [
            "attention/1UBQ_1",
            "structure",
            "trace/attention",
            "trace/activations",
        ] {
            fs::create_dir_all(workspace.join(dir)).unwrap();
        }
        for file in [
            "1UBQ_1_model_1_ptm_relaxed.pdb",
            "1UBQ_1_model_1_ptm_unrelaxed.pdb",
            "1UBQ_1_model_1_ptm_output_dict.pkl",
            "timings.json",
            "attention/1UBQ_1/msa_row_attn_layer47.txt",
            "attention/1UBQ_1/msa_row_attn_layer47.npz",
            "structure/predicted.pdb",
            "trace/attention/layer_000.pt",
            "trace/activations/layer_000.pt",
            "trace/summary.json",
            "notes.md",
        ] {
            fs::write(workspace.join(file), "x").unwrap();
        }
    }

    #[test]
    fn every_produced_file_is_classified() {
        let root = std::env::temp_dir().join(format!("vizfold-produced-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        write_run_outputs(&root);

        let produced = produced_files(&root);
        assert_eq!(produced.len(), 11, "every file, and only files");
        let kinds: Vec<&str> = produced.iter().map(|(_, kind)| *kind).collect();
        for expected in [
            "protein_structure",
            "attention_map",
            "attention_tensor",
            "activation_tensor",
            "model_output_archive",
            "trace_summary",
            "run_metadata",
            // The one file no kind claims is still registered rather than dropped.
            "run_file",
        ] {
            assert!(kinds.contains(&expected), "{expected} was not classified");
        }

        fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn registers_every_produced_file_under_its_kind_idempotently() -> Result<(), DbErr> {
        let db = crate::core::test_support::seeded_db().await?;
        let root = std::env::temp_dir().join(format!("vizfold-register-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let run = run_with_output_root(&db, &root).await?;
        let workspace = root.join(run.id.to_string());
        fs::create_dir_all(&workspace).unwrap();
        write_run_outputs(&workspace);

        let report = register_run_artifacts(&db, run.id).await?;
        assert_eq!(report.get("protein_structure"), Some(&3));
        assert_eq!(report.get("attention_map"), Some(&1));
        assert_eq!(report.get("attention_tensor"), Some(&2));
        assert_eq!(report.get("activation_tensor"), Some(&1));
        assert_eq!(report.get("run_output_directory"), Some(&1));

        let artifacts = register_known_run_artifacts(&db, run.id).await?;
        assert_eq!(artifacts.len(), 12, "eleven files plus the workspace");

        // A second pass adds nothing: a run is registered against what is on disk, not against
        // how many times anyone asked.
        assert!(register_run_artifacts(&db, run.id).await?.is_empty());

        // But a row typed by an older rule converges on the current one rather than staying wrong.
        let structure = artifacts
            .iter()
            .find(|artifact| artifact.storage_uri.ends_with("predicted.pdb"))
            .expect("the ESMFold structure is registered");
        let run_file = crate::core::entities::artifact_types::Entity::find()
            .filter(crate::core::entities::artifact_types::Column::Slug.eq("run_file"))
            .one(&db)
            .await?
            .expect("run_file is seeded");
        crate::core::services::artifacts::reclassify_artifact(
            &db,
            structure.id,
            run_file.id,
            "{}".into(),
        )
        .await?;
        let repaired = register_run_artifacts(&db, run.id).await?;
        assert_eq!(repaired.get("protein_structure"), Some(&1));

        // The instance carries what its name encodes, so nothing downstream parses it again.
        let attention = artifacts
            .iter()
            .find(|artifact| artifact.storage_uri.ends_with("msa_row_attn_layer47.txt"))
            .expect("the attention dump is registered");
        let metadata: serde_json::Value = serde_json::from_str(&attention.metadata_json).unwrap();
        assert_eq!(metadata["target"], "1UBQ_1");
        assert_eq!(metadata["layer"], 47);
        assert_eq!(metadata["attention"], "msa_row");
        assert_eq!(attention.format, "txt");

        fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[tokio::test]
    async fn a_run_that_wrote_nothing_registers_nothing() -> Result<(), DbErr> {
        let db = crate::core::test_support::seeded_db().await?;
        let root = std::env::temp_dir().join(format!("vizfold-empty-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let run = run_with_output_root(&db, &root).await?;

        assert!(register_run_artifacts(&db, run.id).await?.is_empty());
        assert!(register_known_run_artifacts(&db, run.id).await?.is_empty());
        Ok(())
    }
}
