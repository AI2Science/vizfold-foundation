use sea_orm::{
    ActiveModelTrait, ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter, Set,
};

use crate::core::entities::artifacts;

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
    run_id: i32,
    kind: &str,
    uri: &str,
    metadata_json: Option<&str>,
) -> Result<artifacts::Model, DbErr> {
    artifacts::ActiveModel {
        run_id: Set(run_id),
        kind: Set(kind.to_owned()),
        uri: Set(uri.to_owned()),
        metadata_json: Set(metadata_json.map(str::to_owned)),
        ..Default::default()
    }
    .insert(db)
    .await
}
