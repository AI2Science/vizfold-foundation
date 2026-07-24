use sea_orm::{
    ActiveModelTrait, ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter, Set,
};

use crate::core::entities::{artifact_types, artifacts};

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
    artifacts::Entity::find()
        .filter(artifacts::Column::RunId.eq(run_id))
        .all(db)
        .await
}

pub async fn record_artifact_manifest_entry(
    db: &DatabaseConnection,
    input: RecordArtifactInput,
) -> Result<artifacts::Model, DbErr> {
    require_json_object("artifact metadata", &input.metadata_json)?;

    artifacts::ActiveModel {
        run_id: Set(input.run_id),
        artifact_type_id: Set(input.artifact_type_id),
        format: Set(input.format),
        storage_uri: Set(input.storage_uri),
        metadata_json: Set(input.metadata_json),
        ..Default::default()
    }
    .insert(db)
    .await
}

pub async fn record_artifact_manifest_entry_by_type_slug(
    db: &DatabaseConnection,
    input: RecordArtifactByTypeSlugInput,
) -> Result<artifacts::Model, DbErr> {
    let artifact_type = artifact_types::Entity::find()
        .filter(artifact_types::Column::Slug.eq(&input.artifact_type_slug))
        .one(db)
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
