use sea_orm::{
    ActiveModelTrait, ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter, Set,
};

use crate::core::entities::artifacts;

use super::validation::require_json_object;

#[derive(Clone, Debug)]
pub struct RecordArtifactInput {
    pub run_id: i32,
    pub artifact_type_id: i32,
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

/// Move an already-recorded artifact onto the kind the classifier now says it is. A run keeps its
/// row -- the file on disk has not changed -- but what the catalog calls it can.
pub async fn reclassify_artifact(
    db: &DatabaseConnection,
    artifact_id: i32,
    artifact_type_id: i32,
    metadata_json: String,
) -> Result<artifacts::Model, DbErr> {
    require_json_object("artifact metadata", &metadata_json)?;

    let mut artifact: artifacts::ActiveModel = artifacts::Entity::find_by_id(artifact_id)
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom(format!("artifact {artifact_id} does not exist")))?
        .into();
    artifact.artifact_type_id = Set(artifact_type_id);
    artifact.metadata_json = Set(metadata_json);
    artifact.update(db).await
}
