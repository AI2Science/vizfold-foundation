use sea_orm::{DatabaseConnection, DbErr};

use crate::core::{entities::model_backends, repositories};

pub async fn list_model_backends(
    db: &DatabaseConnection,
) -> Result<Vec<model_backends::Model>, DbErr> {
    repositories::model_backends::list(db).await
}

pub async fn create_model_backend(
    db: &DatabaseConnection,
    slug: &str,
    label: &str,
    summary: &str,
    capabilities_json: &str,
) -> Result<model_backends::Model, DbErr> {
    repositories::model_backends::create(db, slug, label, summary, capabilities_json).await
}
