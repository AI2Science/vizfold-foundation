use sea_orm::{
    ActiveModelTrait, ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter, Set,
};

use crate::core::{
    entities::execution_targets, services::execution_targets::RegisterExecutionTargetInput,
};

pub async fn list(db: &DatabaseConnection) -> Result<Vec<execution_targets::Model>, DbErr> {
    execution_targets::Entity::find().all(db).await
}

pub async fn find_by_id(
    db: &DatabaseConnection,
    id: i32,
) -> Result<Option<execution_targets::Model>, DbErr> {
    execution_targets::Entity::find_by_id(id).one(db).await
}

pub async fn create(
    db: &DatabaseConnection,
    input: RegisterExecutionTargetInput,
) -> Result<execution_targets::Model, DbErr> {
    execution_targets::ActiveModel {
        slug: Set(input.slug),
        target_type: Set(input.target_type),
        description: Set(input.description),
        available_resources_json: Set(input.available_resources_json),
        ..Default::default()
    }
    .insert(db)
    .await
}

pub async fn find_by_slug(
    db: &DatabaseConnection,
    slug: &str,
) -> Result<Option<execution_targets::Model>, DbErr> {
    execution_targets::Entity::find()
        .filter(execution_targets::Column::Slug.eq(slug))
        .one(db)
        .await
}
