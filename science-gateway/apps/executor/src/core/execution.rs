use sea_orm::{DatabaseConnection, DbErr};

use crate::core::{db, services};

pub struct ExecutionCore {
    db: DatabaseConnection,
}

impl ExecutionCore {
    pub async fn bootstrap() -> Result<Self, DbErr> {
        let db = db::connect_and_migrate().await?;
        Ok(Self { db })
    }

    pub async fn check_readiness(&self) -> Result<(), DbErr> {
        let _ = services::model_backends::list_model_backends(&self.db).await?;
        Ok(())
    }

    pub fn db(&self) -> &DatabaseConnection {
        &self.db
    }
}
