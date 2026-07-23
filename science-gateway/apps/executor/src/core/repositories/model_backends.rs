use sea_orm::{
    ActiveModelTrait, ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter, Set,
};

use crate::core::{entities::model_backends, services::model_backends::RegisterModelBackendInput};

pub async fn list(db: &DatabaseConnection) -> Result<Vec<model_backends::Model>, DbErr> {
    model_backends::Entity::find().all(db).await
}

pub async fn find_by_id(
    db: &DatabaseConnection,
    id: i32,
) -> Result<Option<model_backends::Model>, DbErr> {
    model_backends::Entity::find_by_id(id).one(db).await
}

pub async fn find_by_slug(
    db: &DatabaseConnection,
    slug: &str,
) -> Result<Option<model_backends::Model>, DbErr> {
    model_backends::Entity::find()
        .filter(model_backends::Column::Slug.eq(slug))
        .one(db)
        .await
}

pub async fn create(
    db: &DatabaseConnection,
    input: RegisterModelBackendInput,
) -> Result<model_backends::Model, DbErr> {
    model_backends::ActiveModel {
        slug: Set(input.slug),
        label: Set(input.label),
        version: Set(input.version),
        description: Set(input.description),
        artifact_capabilities_json: Set(input.artifact_capabilities_json),
        parameter_schema_json: Set(input.parameter_schema_json),
        ..Default::default()
    }
    .insert(db)
    .await
}
