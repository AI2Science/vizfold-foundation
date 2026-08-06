use sea_orm::{ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter};
use serde_json::json;

use crate::core::{
    artifact_kinds, config,
    entities::{artifact_types, execution_targets, model_backends, model_invocation_profiles},
    services,
};

pub async fn seed_defaults(db: &DatabaseConnection) -> Result<(), DbErr> {
    // The catalog is `artifact_kinds::KINDS`, which is also what classifies a produced file, so
    // the kinds a run can be registered as and the kinds this table holds cannot drift apart.
    for kind in artifact_kinds::KINDS {
        let input = services::artifact_types::RegisterArtifactTypeInput {
            slug: kind.slug.into(),
            label: kind.label.into(),
            default_format: kind.default_format.into(),
            display_mode: kind.display_mode.into(),
            viewer_kind: kind.viewer_kind.into(),
            description: kind.description.into(),
            metadata_schema_json: kind.metadata_schema_json.into(),
        };
        // The slug identifies a kind; everything else is the current definition of it. An install
        // upgraded in place holds rows written by an older catalog, and the dashboard reads its
        // viewer and display mode off those rows -- so an existing kind is brought current, not
        // left as whichever version first wrote it.
        match artifact_types::Entity::find()
            .filter(artifact_types::Column::Slug.eq(kind.slug))
            .one(db)
            .await?
        {
            Some(existing) if catalog_differs(&existing, kind) => {
                services::artifact_types::update_artifact_type(db, existing.id, input).await?;
            }
            Some(_) => {}
            None => {
                services::artifact_types::register_artifact_type(db, input).await?;
            }
        }
    }

    seed_backend(
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
        "scripts/openfold/run_pretrained_openfold.py",
    )
    .await?;

    seed_backend(
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
            services::execution_targets::RegisterExecutionTargetInput {
        slug: "local-esmfold".into(),
        target_type: "local".into(),
        description: Some("Local ESMFold subprocess execution target.".into()),
        available_resources_json:
            r#"{"type":"object","properties":{"model_device":{"type":"string","enum":["cpu","cuda","cuda:0"],"default":"cuda:0","cli_flag":"--device"}}}"#
                .into(),
    },
        "scripts/esmfold/run_pretrained_esmf.py",
    )
    .await?;

    Ok(())
}

/// Register a backend, its local execution target, and the profile binding them -- each only where
/// it is missing, since this runs on every connect. What the backends differ by is the arguments.
async fn seed_backend(
    db: &DatabaseConnection,
    backend: services::model_backends::RegisterModelBackendInput,
    target: services::execution_targets::RegisterExecutionTargetInput,
    script: &str,
) -> Result<(), DbErr> {
    let (backend_slug, target_slug) = (backend.slug.clone(), target.slug.clone());

    if model_backends::Entity::find()
        .filter(model_backends::Column::Slug.eq(&*backend_slug))
        .one(db)
        .await?
        .is_none()
    {
        services::model_backends::register_model_backend(db, backend).await?;
    }

    if execution_targets::Entity::find()
        .filter(execution_targets::Column::Slug.eq(&*target_slug))
        .one(db)
        .await?
        .is_none()
    {
        services::execution_targets::register_execution_target(db, target).await?;
    }

    let backend = model_backends::Entity::find()
        .filter(model_backends::Column::Slug.eq(&*backend_slug))
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom(format!("seeded {backend_slug} model backend is missing")))?;
    let target = execution_targets::Entity::find()
        .filter(execution_targets::Column::Slug.eq(&*target_slug))
        .one(db)
        .await?
        .ok_or_else(|| {
            DbErr::Custom(format!("seeded {target_slug} execution target is missing"))
        })?;

    // The profile carries paths that move with the config, so an existing one is brought current
    // rather than left at whatever prefix it was first seeded against.
    let config_json = local_config_json(script);
    match model_invocation_profiles::Entity::find()
        .filter(model_invocation_profiles::Column::ModelBackendId.eq(backend.id))
        .filter(model_invocation_profiles::Column::ExecutionTargetId.eq(target.id))
        .one(db)
        .await?
    {
        Some(profile) if profile.config_json != config_json => {
            services::model_invocation_profiles::update_config(db, profile.id, config_json).await?;
        }
        Some(_) => {}
        None => {
            services::model_invocation_profiles::register_model_invocation_profile(
                db,
                services::model_invocation_profiles::RegisterModelInvocationProfileInput {
                    model_backend_id: backend.id,
                    execution_target_id: target.id,
                    invocation_kind: "local_subprocess".into(),
                    config_json,
                },
            )
            .await?;
        }
    }

    Ok(())
}

fn local_config_json(script: &str) -> String {
    json!({
        "program": "python3",
        // The entrypoints import by module, so working_dir only has to make examples/ resolve.
        "script": script,
        "working_dir": config::openfold_home(),
        "output_location": config::prefix().join("runs"),
    })
    .to_string()
}

/// Whether a seeded row still says what the code says.
fn catalog_differs(row: &artifact_types::Model, kind: &artifact_kinds::ArtifactKind) -> bool {
    row.label != kind.label
        || row.default_format != kind.default_format
        || row.display_mode != kind.display_mode
        || row.viewer_kind != kind.viewer_kind
        || row.description != kind.description
        || row.metadata_schema_json != kind.metadata_schema_json
}
