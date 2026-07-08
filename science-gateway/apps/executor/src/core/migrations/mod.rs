mod m20260707_000001_create_model_backends_table;
mod m20260707_000002_create_runs_table;
mod m20260707_000003_create_artifacts_table;

pub use sea_orm_migration::prelude::MigratorTrait;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn sea_orm_migration::MigrationTrait>> {
        vec![
            Box::new(m20260707_000001_create_model_backends_table::Migration),
            Box::new(m20260707_000002_create_runs_table::Migration),
            Box::new(m20260707_000003_create_artifacts_table::Migration),
        ]
    }
}
