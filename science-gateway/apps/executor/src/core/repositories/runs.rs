use sea_orm::{ActiveModelTrait, ActiveValue::Set, DatabaseConnection, DbErr, EntityTrait};

use crate::core::{
    entities::runs,
    services::runs::{SubmitRunInput, UpdateRunStatusInput},
};

pub async fn list(db: &DatabaseConnection) -> Result<Vec<runs::Model>, DbErr> {
    runs::Entity::find().all(db).await
}

pub async fn find_by_id(db: &DatabaseConnection, id: i32) -> Result<Option<runs::Model>, DbErr> {
    runs::Entity::find_by_id(id).one(db).await
}

pub async fn create(db: &DatabaseConnection, input: SubmitRunInput) -> Result<runs::Model, DbErr> {
    runs::ActiveModel {
        model_backend_id: Set(input.model_backend_id),
        execution_target_id: Set(input.execution_target_id),
        invocation_profile_id: Set(input.invocation_profile_id),
        status: Set(input.status),
        input_id: Set(input.input_id),
        input_sequence: Set(input.input_sequence),
        model_parameters_json: Set(input.model_parameters_json),
        execution_parameters_json: Set(input.execution_parameters_json),
        ..Default::default()
    }
    .insert(db)
    .await
}

pub async fn update_status(
    db: &DatabaseConnection,
    run_id: i32,
    update: UpdateRunStatusInput,
) -> Result<runs::Model, DbErr> {
    let model = runs::Entity::find_by_id(run_id)
        .one(db)
        .await?
        .ok_or_else(|| DbErr::Custom("run does not exist".into()))?;

    let mut active_model: runs::ActiveModel = model.into();
    active_model.status = Set(update.status);

    if let Some(started_at) = update.started_at {
        active_model.started_at = Set(started_at);
    }

    if let Some(completed_at) = update.completed_at {
        active_model.completed_at = Set(completed_at);
    }

    if let Some(error_message) = update.error_message {
        active_model.error_message = Set(error_message);
    }

    active_model.update(db).await
}
