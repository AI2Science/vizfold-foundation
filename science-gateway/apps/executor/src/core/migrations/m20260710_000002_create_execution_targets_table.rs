use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(ExecutionTargets::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(ExecutionTargets::Id)
                            .integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(
                        ColumnDef::new(ExecutionTargets::Slug)
                            .string()
                            .not_null()
                            .unique_key(),
                    )
                    .col(
                        ColumnDef::new(ExecutionTargets::TargetType)
                            .string()
                            .not_null(),
                    )
                    .col(ColumnDef::new(ExecutionTargets::Description).text())
                    .col(
                        ColumnDef::new(ExecutionTargets::AvailableResourcesJson)
                            .text()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ExecutionTargets::CreatedAt)
                            .timestamp()
                            .not_null()
                            .default(Expr::current_timestamp()),
                    )
                    .col(
                        ColumnDef::new(ExecutionTargets::UpdatedAt)
                            .timestamp()
                            .not_null()
                            .default(Expr::current_timestamp()),
                    )
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(ExecutionTargets::Table).to_owned())
            .await
    }
}

#[derive(DeriveIden)]
enum ExecutionTargets {
    Table,
    Id,
    Slug,
    TargetType,
    Description,
    AvailableResourcesJson,
    CreatedAt,
    UpdatedAt,
}
