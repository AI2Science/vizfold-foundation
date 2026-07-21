use sea_orm::{DatabaseConnection, DbErr};

use crate::core::{entities::model_invocation_profiles, repositories};

use super::validation::require_json_object;

#[derive(Clone, Debug)]
pub struct RegisterModelInvocationProfileInput {
    pub model_backend_id: i32,
    pub execution_target_id: i32,
    pub invocation_kind: String,
    pub config_json: String,
}

pub async fn list_model_invocation_profiles(
    db: &DatabaseConnection,
) -> Result<Vec<model_invocation_profiles::Model>, DbErr> {
    repositories::model_invocation_profiles::list(db).await
}

pub async fn register_model_invocation_profile(
    db: &DatabaseConnection,
    input: RegisterModelInvocationProfileInput,
) -> Result<model_invocation_profiles::Model, DbErr> {
    let _backend = repositories::model_backends::find_by_id(db, input.model_backend_id)
        .await?
        .ok_or_else(|| DbErr::Custom("model backend does not exist".into()))?;
    let _target = repositories::execution_targets::find_by_id(db, input.execution_target_id)
        .await?
        .ok_or_else(|| DbErr::Custom("execution target does not exist".into()))?;

    require_json_object("model invocation profile config", &input.config_json)?;
    repositories::model_invocation_profiles::create(db, input).await
}
