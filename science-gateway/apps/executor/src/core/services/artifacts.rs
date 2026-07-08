use sea_orm::{DatabaseConnection, DbErr};

use crate::core::{entities::artifacts, repositories};

pub async fn list_artifacts_for_run(
    db: &DatabaseConnection,
    run_id: i32,
) -> Result<Vec<artifacts::Model>, DbErr> {
    repositories::artifacts::list_by_run(db, run_id).await
}

pub async fn create_artifact(
    db: &DatabaseConnection,
    run_id: i32,
    kind: &str,
    uri: &str,
    metadata_json: Option<&str>,
) -> Result<artifacts::Model, DbErr> {
    repositories::artifacts::create(db, run_id, kind, uri, metadata_json).await
}
