use sea_orm::{ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter};
use serde_json::json;

use crate::core::{
    config,
    entities::{artifact_types, execution_targets, model_backends, model_invocation_profiles},
    services,
};

pub async fn seed_defaults(db: &DatabaseConnection) -> Result<(), DbErr> {
    for (slug, label, default_format, display_mode, viewer_kind, description) in [
        (
            "arc_diagram",
            "Arc diagram",
            "png",
            "native",
            "image",
            "Static arc diagram image.",
        ),
        (
            "attention_heatmap",
            "Attention heatmap",
            "png",
            "native",
            "image",
            "Static attention heatmap image.",
        ),
        (
            "combined_3d_arc_panel",
            "Combined 3D and arc panel",
            "png",
            "native",
            "image",
            "Static combined visualization panel.",
        ),
        (
            "pymol_overlay_render",
            "PyMOL 3D overlay render",
            "png",
            "native",
            "image",
            "Static PyMOL-rendered overlay image.",
        ),
        (
            "protein_structure",
            "Protein structure",
            "pdb",
            "embedded",
            "ngl_viewer",
            "Protein structure file suitable for browser-based 3D viewing.",
        ),
        (
            "pymol_session",
            "PyMOL session",
            "pse",
            "download",
            "download_link",
            "PyMOL session file for local use.",
        ),
        (
            "attention_trace_text",
            "Attention trace text",
            "txt",
            "download",
            "download_link",
            "Raw attention trace text output.",
        ),
        (
            "activation_arrays",
            "Activation arrays",
            "npz",
            "download",
            "download_link",
            "NumPy activation arrays.",
        ),
        (
            "trace_archive",
            "Trace archive",
            "zip",
            "download",
            "download_link",
            "Full run trace archive.",
        ),
        (
            "manifest",
            "Manifest",
            "json",
            "internal",
            "viewer_registry",
            "Internal manifest used to describe produced artifacts.",
        ),
        (
            "streamlit_app",
            "Streamlit app",
            "url",
            "embedded",
            "iframe",
            "Optional live Streamlit app URL when available.",
        ),
        (
            "run_output_directory",
            "Run output directory",
            "directory",
            "download",
            "directory_link",
            "Directory containing outputs produced by a run.",
        ),
        (
            "attention_output_directory",
            "Attention output directory",
            "directory",
            "download",
            "directory_link",
            "Directory containing attention-map outputs produced by a run.",
        ),
    ] {
        if artifact_types::Entity::find()
            .filter(artifact_types::Column::Slug.eq(slug))
            .one(db)
            .await?
            .is_none()
        {
            services::artifact_types::register_artifact_type(
                db,
                services::artifact_types::RegisterArtifactTypeInput {
                    slug: slug.into(),
                    label: label.into(),
                    default_format: default_format.into(),
                    display_mode: display_mode.into(),
                    viewer_kind: viewer_kind.into(),
                    description: description.into(),
                    metadata_schema_json: "{}".into(),
                },
            )
            .await?;
        }
    }

    if model_backends::Entity::find()
        .filter(model_backends::Column::Slug.eq("openfold"))
        .one(db)
        .await?
        .is_none()
    {
        services::model_backends::register_model_backend(
            db,
            services::model_backends::RegisterModelBackendInput {
                slug: "openfold".into(),
                label: "OpenFold".into(),
                version: Some("demo".into()),
                description: Some("OpenFold backend placeholder for executor core development.".into()),
                artifact_capabilities_json:
                    r#"{"structure":{"formats":["pdb","cif"],"required":true},"confidence_metrics":{"formats":["json"],"required":false}}"#
                        .into(),
                parameter_schema_json:
                    r#"{"type":"object","properties":{"config_preset":{"type":"string","default":"model_1_ptm","cli_flag":"--config_preset"},"fasta_dir":{"type":"path","source":"execution_parameters","parameter":"fasta_dir","positional":true,"position":1},"template_mmcif_dir":{"type":"path","source":"data_dir","relative_path":"pdb_mmcif/mmcif_files","positional":true,"position":2},"uniref90_database_path":{"type":"path","source":"data_dir","relative_path":"uniref90/uniref90.fasta","cli_flag":"--uniref90_database_path"},"mgnify_database_path":{"type":"path","source":"data_dir","relative_path":"mgnify/mgy_clusters_2022_05.fa","cli_flag":"--mgnify_database_path"},"pdb70_database_path":{"type":"path","source":"data_dir","relative_path":"pdb70/pdb70","cli_flag":"--pdb70_database_path"},"uniclust30_database_path":{"type":"path","source":"data_dir","relative_path":"uniclust30/uniclust30_2018_08/uniclust30_2018_08","cli_flag":"--uniclust30_database_path"},"bfd_database_path":{"type":"path","source":"data_dir","relative_path":"bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt","cli_flag":"--bfd_database_path"},"output_dir":{"type":"path","source":"run_output_workspace","cli_flag":"--output_dir"},"attn_map_dir":{"type":"path","source":"run_output_workspace","relative_path":"attention","cli_flag":"--attn_map_dir"},"save_outputs":{"type":"boolean","cli_flag":"--save_outputs"},"demo_attn":{"type":"boolean","cli_flag":"--demo_attn"},"num_recycles_save":{"type":"integer","cli_flag":"--num_recycles_save"}}}"#
                        .into(),
            },
        )
        .await?;
    }

    if execution_targets::Entity::find()
        .filter(execution_targets::Column::Slug.eq("local-openfold"))
        .one(db)
        .await?
        .is_none()
    {
        services::execution_targets::register_execution_target(
            db,
            services::execution_targets::RegisterExecutionTargetInput {
                slug: "local-openfold".into(),
                target_type: "local".into(),
                description: Some(
                    "Local OpenFold subprocess execution target for demo/development.".into(),
                ),
                available_resources_json:
                    r#"{"type":"object","properties":{"model_device":{"type":"string","enum":["cpu","cuda:0"],"default":"cuda:0","cli_flag":"--model_device"},"cpus":{"type":"integer","minimum":1,"maximum":14,"cli_flag":"--cpus"}}}"#
                        .into(),
            },
        )
        .await?;
    }

    let backend = model_backends::Entity::find()
        .filter(model_backends::Column::Slug.eq("openfold"))
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("seeded OpenFold model backend is missing".into()))?;

    let openfold_target = execution_targets::Entity::find()
        .filter(execution_targets::Column::Slug.eq("local-openfold"))
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("seeded local OpenFold execution target is missing".into()))?;

    let local_openfold_config = local_openfold_config_json();
    if let Some(profile) = model_invocation_profiles::Entity::find()
        .filter(model_invocation_profiles::Column::ModelBackendId.eq(backend.id))
        .filter(model_invocation_profiles::Column::ExecutionTargetId.eq(openfold_target.id))
        .one(db)
        .await?
    {
        if profile.config_json != local_openfold_config {
            services::model_invocation_profiles::update_config(
                db,
                profile.id,
                local_openfold_config,
            )
            .await?;
        }
    } else {
        services::model_invocation_profiles::register_model_invocation_profile(
            db,
            services::model_invocation_profiles::RegisterModelInvocationProfileInput {
                model_backend_id: backend.id,
                execution_target_id: openfold_target.id,
                invocation_kind: "local_subprocess".into(),
                config_json: local_openfold_config,
            },
        )
        .await?;
    }

    seed_esmfold(db).await?;

    Ok(())
}

/// ESMFold catalog rows: the backend's CLI schema, a local execution target, and a
/// local_subprocess profile pointing at `run_pretrained_esmf.py`. Single-sequence, so no
/// AlphaFold2 database schema -- just the model/trace flags and a `--device`.
async fn seed_esmfold(db: &DatabaseConnection) -> Result<(), DbErr> {
    if model_backends::Entity::find()
        .filter(model_backends::Column::Slug.eq("esmfold"))
        .one(db)
        .await?
        .is_none()
    {
        services::model_backends::register_model_backend(
            db,
            services::model_backends::RegisterModelBackendInput {
                slug: "esmfold".into(),
                label: "ESMFold".into(),
                version: Some("esmfold_v1".into()),
                description: Some("ESMFold backend (HuggingFace EsmForProteinFolding).".into()),
                artifact_capabilities_json:
                    r#"{"structure":{"formats":["pdb"],"required":true}}"#.into(),
                parameter_schema_json:
                    r#"{"type":"object","properties":{"fasta":{"type":"path","source":"execution_parameters","parameter":"fasta","cli_flag":"--fasta"},"out":{"type":"path","source":"run_output_workspace","cli_flag":"--out"},"model":{"type":"string","default":"facebook/esmfold_v1","cli_flag":"--model"},"dtype":{"type":"string","default":"float32","cli_flag":"--dtype"},"trace_mode":{"type":"string","default":"attention+activations","cli_flag":"--trace_mode"},"layers":{"type":"string","default":"all","cli_flag":"--layers"},"heads":{"type":"string","default":"all","cli_flag":"--heads"},"top_k":{"type":"integer","default":50,"cli_flag":"--top_k"},"save_fp16":{"type":"boolean","cli_flag":"--save_fp16"},"structure_traces":{"type":"boolean","cli_flag":"--structure_traces"}}}"#
                        .into(),
            },
        )
        .await?;
    }

    if execution_targets::Entity::find()
        .filter(execution_targets::Column::Slug.eq("local-esmfold"))
        .one(db)
        .await?
        .is_none()
    {
        services::execution_targets::register_execution_target(
            db,
            services::execution_targets::RegisterExecutionTargetInput {
                slug: "local-esmfold".into(),
                target_type: "local".into(),
                description: Some("Local ESMFold subprocess execution target.".into()),
                available_resources_json:
                    r#"{"type":"object","properties":{"model_device":{"type":"string","enum":["cpu","cuda","cuda:0"],"default":"cuda:0","cli_flag":"--device"}}}"#
                        .into(),
            },
        )
        .await?;
    }

    let backend = model_backends::Entity::find()
        .filter(model_backends::Column::Slug.eq("esmfold"))
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("seeded ESMFold model backend is missing".into()))?;
    let target = execution_targets::Entity::find()
        .filter(execution_targets::Column::Slug.eq("local-esmfold"))
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("seeded local ESMFold execution target is missing".into()))?;

    let config = local_esmfold_config_json();
    if let Some(profile) = model_invocation_profiles::Entity::find()
        .filter(model_invocation_profiles::Column::ModelBackendId.eq(backend.id))
        .filter(model_invocation_profiles::Column::ExecutionTargetId.eq(target.id))
        .one(db)
        .await?
    {
        if profile.config_json != config {
            services::model_invocation_profiles::update_config(db, profile.id, config).await?;
        }
    } else {
        services::model_invocation_profiles::register_model_invocation_profile(
            db,
            services::model_invocation_profiles::RegisterModelInvocationProfileInput {
                model_backend_id: backend.id,
                execution_target_id: target.id,
                invocation_kind: "local_subprocess".into(),
                config_json: config,
            },
        )
        .await?;
    }

    Ok(())
}

fn local_openfold_config_json() -> String {
    json!({
        "program": "python3",
        // The backend subtree is in the script path, not working_dir: python adds the script's
        // directory to sys.path (so `import openfold`/`scripts` resolve), while working_dir stays
        // the checkout root so relative run inputs (examples/, ...) resolve there.
        "script": "backends/openfold/run_pretrained_openfold.py",
        "working_dir": config::openfold_home(),
        "output_location": config::prefix().join("runs"),
    })
    .to_string()
}

fn local_esmfold_config_json() -> String {
    json!({
        "program": "python3",
        "script": "backends/esmfold/run_pretrained_esmf.py",
        "working_dir": config::openfold_home(),
        "output_location": config::prefix().join("runs"),
    })
    .to_string()
}
