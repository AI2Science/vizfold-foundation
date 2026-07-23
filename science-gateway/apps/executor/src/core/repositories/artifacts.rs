use sea_orm::{
    ActiveModelTrait, ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter, Set,
};

use crate::core::{entities::artifacts, services::artifacts::RecordArtifactInput};

pub async fn list_by_run(
    db: &DatabaseConnection,
    run_id: i32,
) -> Result<Vec<artifacts::Model>, DbErr> {
    artifacts::Entity::find()
        .filter(artifacts::Column::RunId.eq(run_id))
        .all(db)
        .await
}

pub async fn create(
    db: &DatabaseConnection,
    input: RecordArtifactInput,
) -> Result<artifacts::Model, DbErr> {
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
