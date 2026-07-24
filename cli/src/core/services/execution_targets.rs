use sea_orm::{ActiveModelTrait, DatabaseConnection, DbErr, EntityTrait, Set};

use crate::core::entities::execution_targets;

use super::validation::require_json_object;

#[derive(Clone, Debug)]
pub struct RegisterExecutionTargetInput {
    pub slug: String,
    pub target_type: String,
    pub description: Option<String>,
    pub available_resources_json: String,
}

pub async fn list_execution_targets(
    db: &DatabaseConnection,
) -> Result<Vec<execution_targets::Model>, DbErr> {
    execution_targets::Entity::find().all(db).await
}

pub async fn register_execution_target(
    db: &DatabaseConnection,
    input: RegisterExecutionTargetInput,
) -> Result<execution_targets::Model, DbErr> {
    require_json_object(
        "execution target available_resources",
        &input.available_resources_json,
    )?;

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
