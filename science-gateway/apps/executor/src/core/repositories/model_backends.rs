use sea_orm::{ActiveModelTrait, DatabaseConnection, DbErr, EntityTrait, Set};

use crate::core::entities::model_backends;

pub async fn list(db: &DatabaseConnection) -> Result<Vec<model_backends::Model>, DbErr> {
    model_backends::Entity::find().all(db).await
}

pub async fn create(
    db: &DatabaseConnection,
    slug: &str,
    label: &str,
    summary: &str,
    capabilities_json: &str,
) -> Result<model_backends::Model, DbErr> {
    model_backends::ActiveModel {
        slug: Set(slug.to_owned()),
        label: Set(label.to_owned()),
        summary: Set(summary.to_owned()),
        capabilities_json: Set(capabilities_json.to_owned()),
        ..Default::default()
    }
    .insert(db)
    .await
}
