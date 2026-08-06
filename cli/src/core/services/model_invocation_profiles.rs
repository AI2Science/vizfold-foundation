use chrono::Utc;
use sea_orm::{ActiveModelTrait, DatabaseConnection, DbErr, EntityTrait, Set};

use crate::core::entities::model_invocation_profiles;

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
    model_invocation_profiles::Entity::find().all(db).await
}

pub async fn register_model_invocation_profile(
    db: &DatabaseConnection,
    input: RegisterModelInvocationProfileInput,
) -> Result<model_invocation_profiles::Model, DbErr> {
    let _backend = super::require_backend(db, input.model_backend_id).await?;
    let _target = super::require_target(db, input.execution_target_id).await?;

    require_json_object("model invocation profile config", &input.config_json)?;
    model_invocation_profiles::ActiveModel {
        model_backend_id: Set(input.model_backend_id),
        execution_target_id: Set(input.execution_target_id),
        invocation_kind: Set(input.invocation_kind),
        config_json: Set(input.config_json),
        ..Default::default()
    }
    .insert(db)
    .await
}

pub async fn update_config(
    db: &DatabaseConnection,
    id: i32,
    config_json: String,
) -> Result<model_invocation_profiles::Model, DbErr> {
    let model = super::require_profile(db, id).await?;
    let mut active_model: model_invocation_profiles::ActiveModel = model.into();
    active_model.config_json = Set(config_json);
    active_model.updated_at = Set(Utc::now());
    active_model.update(db).await
}
