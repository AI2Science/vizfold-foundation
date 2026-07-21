use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(ModelInvocationProfiles::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(ModelInvocationProfiles::Id)
                            .integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(
                        ColumnDef::new(ModelInvocationProfiles::ModelBackendId)
                            .integer()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ModelInvocationProfiles::ExecutionTargetId)
                            .integer()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ModelInvocationProfiles::InvocationKind)
                            .string()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ModelInvocationProfiles::ConfigJson)
                            .text()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ModelInvocationProfiles::CreatedAt)
                            .timestamp()
                            .not_null()
                            .default(Expr::current_timestamp()),
                    )
                    .col(
                        ColumnDef::new(ModelInvocationProfiles::UpdatedAt)
                            .timestamp()
                            .not_null()
                            .default(Expr::current_timestamp()),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_model_invocation_profiles_model_backend_id")
                            .from(
                                ModelInvocationProfiles::Table,
                                ModelInvocationProfiles::ModelBackendId,
                            )
                            .to(ModelBackends::Table, ModelBackends::Id)
                            .on_delete(ForeignKeyAction::Restrict)
                            .on_update(ForeignKeyAction::Cascade),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_model_invocation_profiles_execution_target_id")
                            .from(
                                ModelInvocationProfiles::Table,
                                ModelInvocationProfiles::ExecutionTargetId,
                            )
                            .to(ExecutionTargets::Table, ExecutionTargets::Id)
                            .on_delete(ForeignKeyAction::Restrict)
                            .on_update(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(
                Table::drop()
                    .table(ModelInvocationProfiles::Table)
                    .to_owned(),
            )
            .await
    }
}

#[derive(DeriveIden)]
enum ModelInvocationProfiles {
    Table,
    Id,
    ModelBackendId,
    ExecutionTargetId,
    InvocationKind,
    ConfigJson,
    CreatedAt,
    UpdatedAt,
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
