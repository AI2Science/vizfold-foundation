use sea_orm::{ActiveModelTrait, DatabaseConnection, DbErr, EntityTrait, Set};

use crate::core::entities::runs;

pub async fn list(db: &DatabaseConnection) -> Result<Vec<runs::Model>, DbErr> {
    runs::Entity::find().all(db).await
}

pub async fn create(
    db: &DatabaseConnection,
    job_name: &str,
    input_text: &str,
    status: &str,
    output_json: Option<&str>,
    model_backend_id: i32,
) -> Result<runs::Model, DbErr> {
    runs::ActiveModel {
        job_name: Set(job_name.to_owned()),
        input_text: Set(input_text.to_owned()),
        status: Set(status.to_owned()),
        output_json: Set(output_json.map(str::to_owned)),
        model_backend_id: Set(model_backend_id),
        ..Default::default()
    }
    .insert(db)
    .await
}
