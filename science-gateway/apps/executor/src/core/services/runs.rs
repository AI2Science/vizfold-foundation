use sea_orm::{DatabaseConnection, DbErr};

use crate::core::{entities::runs, repositories};

pub async fn list_runs(db: &DatabaseConnection) -> Result<Vec<runs::Model>, DbErr> {
    repositories::runs::list(db).await
}

pub async fn create_run(
    db: &DatabaseConnection,
    job_name: &str,
    input_text: &str,
    status: &str,
    output_json: Option<&str>,
    model_backend_id: i32,
) -> Result<runs::Model, DbErr> {
    repositories::runs::create(db, job_name, input_text, status, output_json, model_backend_id)
        .await
}
