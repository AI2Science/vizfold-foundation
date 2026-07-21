use sea_orm::{DatabaseConnection, DbErr};

use crate::core::{entities::execution_targets, repositories};

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
    repositories::execution_targets::list(db).await
}

pub async fn register_execution_target(
    db: &DatabaseConnection,
    input: RegisterExecutionTargetInput,
) -> Result<execution_targets::Model, DbErr> {
    require_json_object(
        "execution target available_resources",
        &input.available_resources_json,
    )?;

    repositories::execution_targets::create(db, input).await
}
