use sea_orm::{ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter};

use crate::core::{
    entities::{execution_targets, model_backends, model_invocation_profiles},
    services,
};

pub async fn seed_defaults(db: &DatabaseConnection) -> Result<(), DbErr> {
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
                    r#"{"type":"object","properties":{"config_preset":{"type":"string","default":"model_1_ptm","cli_flag":"--config_preset"},"template_mmcif_dir":{"type":"path","source":"data_dir","relative_path":"pdb_mmcif/mmcif_files","positional":true,"position":2},"uniref90_database_path":{"type":"path","source":"data_dir","relative_path":"uniref90/uniref90.fasta","cli_flag":"--uniref90_database_path"},"mgnify_database_path":{"type":"path","source":"data_dir","relative_path":"mgnify/mgy_clusters_2022_05.fa","cli_flag":"--mgnify_database_path"},"pdb70_database_path":{"type":"path","source":"data_dir","relative_path":"pdb70/pdb70","cli_flag":"--pdb70_database_path"},"uniclust30_database_path":{"type":"path","source":"data_dir","relative_path":"uniclust30/uniclust30_2018_08/uniclust30_2018_08","cli_flag":"--uniclust30_database_path"},"bfd_database_path":{"type":"path","source":"data_dir","relative_path":"bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt","cli_flag":"--bfd_database_path"},"save_outputs":{"type":"boolean","cli_flag":"--save_outputs"},"demo_attn":{"type":"boolean","cli_flag":"--demo_attn"},"num_recycles_save":{"type":"integer","cli_flag":"--num_recycles_save"}}}"#
                        .into(),
            },
        )
        .await?;
    }

    if execution_targets::Entity::find()
        .filter(execution_targets::Column::Slug.eq("local-mock"))
        .one(db)
        .await?
        .is_none()
    {
        services::execution_targets::register_execution_target(
            db,
            services::execution_targets::RegisterExecutionTargetInput {
                slug: "local-mock".into(),
                target_type: "local".into(),
                description: Some("Local mock execution target for development and tests.".into()),
                parameter_schema_json:
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
    let target = execution_targets::Entity::find()
        .filter(execution_targets::Column::Slug.eq("local-mock"))
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("seeded local mock execution target is missing".into()))?;

    if model_invocation_profiles::Entity::find()
        .filter(model_invocation_profiles::Column::ModelBackendId.eq(backend.id))
        .filter(model_invocation_profiles::Column::ExecutionTargetId.eq(target.id))
        .one(db)
        .await?
        .is_none()
    {
        services::model_invocation_profiles::register_model_invocation_profile(
            db,
            services::model_invocation_profiles::RegisterModelInvocationProfileInput {
                model_backend_id: backend.id,
                execution_target_id: target.id,
                invocation_kind: "mock".into(),
                config_json: r#"{"mode":"local_mock"}"#.into(),
                parameter_schema_json: r#"{"type":"object","properties":{}}"#.into(),
            },
        )
        .await?;
    }

    Ok(())
}
