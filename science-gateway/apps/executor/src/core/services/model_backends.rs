use sea_orm::{DatabaseConnection, DbErr};

use crate::core::{entities::model_backends, repositories};

use super::validation::require_json_object;

#[derive(Clone, Debug)]
pub struct RegisterModelBackendInput {
    pub slug: String,
    pub label: String,
    pub version: Option<String>,
    pub description: Option<String>,
    pub artifact_capabilities_json: String,
    pub parameter_schema_json: String,
}

pub async fn list_model_backends(
    db: &DatabaseConnection,
) -> Result<Vec<model_backends::Model>, DbErr> {
    repositories::model_backends::list(db).await
}

pub async fn register_model_backend(
    db: &DatabaseConnection,
    input: RegisterModelBackendInput,
) -> Result<model_backends::Model, DbErr> {
    require_json_object(
        "model backend artifact_capabilities",
        &input.artifact_capabilities_json,
    )?;
    require_json_object(
        "model backend parameter_schema",
        &input.parameter_schema_json,
    )?;

    repositories::model_backends::create(db, input).await
}
