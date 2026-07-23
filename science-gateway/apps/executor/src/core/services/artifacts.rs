use sea_orm::{DatabaseConnection, DbErr};

use crate::core::{entities::artifacts, repositories};

use super::validation::require_json_object;

#[derive(Clone, Debug)]
pub struct RecordArtifactInput {
    pub run_id: i32,
    pub artifact_type_id: i32,
    pub format: String,
    pub storage_uri: String,
    pub metadata_json: String,
}

#[derive(Clone, Debug)]
pub struct RecordArtifactByTypeSlugInput {
    pub run_id: i32,
    pub artifact_type_slug: String,
    pub format: String,
    pub storage_uri: String,
    pub metadata_json: String,
}

pub async fn list_artifacts_for_run(
    db: &DatabaseConnection,
    run_id: i32,
) -> Result<Vec<artifacts::Model>, DbErr> {
    repositories::artifacts::list_by_run(db, run_id).await
}

pub async fn record_artifact_manifest_entry(
    db: &DatabaseConnection,
    input: RecordArtifactInput,
) -> Result<artifacts::Model, DbErr> {
    require_json_object("artifact metadata", &input.metadata_json)?;

    repositories::artifacts::create(db, input).await
}

pub async fn record_artifact_manifest_entry_by_type_slug(
    db: &DatabaseConnection,
    input: RecordArtifactByTypeSlugInput,
) -> Result<artifacts::Model, DbErr> {
    let artifact_type = repositories::artifact_types::find_by_slug(db, &input.artifact_type_slug)
        .await?
        .ok_or_else(|| {
            DbErr::Custom(format!(
                "artifact type '{}' was not found",
                input.artifact_type_slug
            ))
        })?;
    record_artifact_manifest_entry(
        db,
        RecordArtifactInput {
            run_id: input.run_id,
            artifact_type_id: artifact_type.id,
            format: input.format,
            storage_uri: input.storage_uri,
            metadata_json: input.metadata_json,
        },
    )
    .await
}
