use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(Runs::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(Runs::Id)
                            .integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(Runs::ModelBackendId).integer().not_null())
                    .col(ColumnDef::new(Runs::ExecutionTargetId).integer().not_null())
                    .col(
                        ColumnDef::new(Runs::InvocationProfileId)
                            .integer()
                            .not_null(),
                    )
                    .col(ColumnDef::new(Runs::Status).string().not_null())
                    .col(ColumnDef::new(Runs::InputSequence).text().not_null())
                    .col(ColumnDef::new(Runs::ModelParametersJson).text().not_null())
                    .col(
                        ColumnDef::new(Runs::ExecutionParametersJson)
                            .text()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(Runs::SubmittedAt)
                            .timestamp()
                            .not_null()
                            .default(Expr::current_timestamp()),
                    )
                    .col(ColumnDef::new(Runs::StartedAt).timestamp())
                    .col(ColumnDef::new(Runs::CompletedAt).timestamp())
                    .col(ColumnDef::new(Runs::ErrorMessage).text())
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_runs_model_backend_id")
                            .from(Runs::Table, Runs::ModelBackendId)
                            .to(ModelBackends::Table, ModelBackends::Id)
                            .on_delete(ForeignKeyAction::Restrict)
                            .on_update(ForeignKeyAction::Cascade),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_runs_execution_target_id")
                            .from(Runs::Table, Runs::ExecutionTargetId)
                            .to(ExecutionTargets::Table, ExecutionTargets::Id)
                            .on_delete(ForeignKeyAction::Restrict)
                            .on_update(ForeignKeyAction::Cascade),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_runs_invocation_profile_id")
                            .from(Runs::Table, Runs::InvocationProfileId)
                            .to(ModelInvocationProfiles::Table, ModelInvocationProfiles::Id)
                            .on_delete(ForeignKeyAction::Restrict)
                            .on_update(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(Runs::Table).to_owned())
            .await
    }
}

#[derive(DeriveIden)]
enum Runs {
    Table,
    Id,
    ModelBackendId,
    ExecutionTargetId,
    InvocationProfileId,
    Status,
    InputSequence,
    ModelParametersJson,
    ExecutionParametersJson,
    SubmittedAt,
    StartedAt,
    CompletedAt,
    ErrorMessage,
}

#[derive(DeriveIden)]
enum ModelBackends {
    Table,
    Id,
}

#[derive(DeriveIden)]
enum ExecutionTargets {
    Table,
    Id,
}

#[derive(DeriveIden)]
enum ModelInvocationProfiles {
    Table,
    Id,
}
