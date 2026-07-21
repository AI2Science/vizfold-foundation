use sea_orm::{ActiveModelTrait, DatabaseConnection, DbErr, EntityTrait, Set};

use crate::core::{
    entities::model_invocation_profiles,
    services::model_invocation_profiles::RegisterModelInvocationProfileInput,
};

pub async fn list(db: &DatabaseConnection) -> Result<Vec<model_invocation_profiles::Model>, DbErr> {
    model_invocation_profiles::Entity::find().all(db).await
}

pub async fn find_by_id(
    db: &DatabaseConnection,
    id: i32,
) -> Result<Option<model_invocation_profiles::Model>, DbErr> {
    model_invocation_profiles::Entity::find_by_id(id)
        .one(db)
        .await
}

pub async fn create(
    db: &DatabaseConnection,
    input: RegisterModelInvocationProfileInput,
) -> Result<model_invocation_profiles::Model, DbErr> {
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
