mod m20260707_000001_create_model_backends_table;
mod m20260707_000002_create_runs_table;
mod m20260707_000003_create_artifacts_table;
mod m20260710_000002_create_execution_targets_table;
mod m20260710_000003_create_model_invocation_profiles_table;
mod m20260716_000004_add_input_id_to_runs_table;

pub use sea_orm_migration::prelude::MigratorTrait;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn sea_orm_migration::MigrationTrait>> {
        vec![
            Box::new(m20260707_000001_create_model_backends_table::Migration),
            Box::new(m20260710_000002_create_execution_targets_table::Migration),
            Box::new(m20260710_000003_create_model_invocation_profiles_table::Migration),
            Box::new(m20260707_000002_create_runs_table::Migration),
            Box::new(m20260707_000003_create_artifacts_table::Migration),
            Box::new(m20260716_000004_add_input_id_to_runs_table::Migration),
        ]
    }
}
