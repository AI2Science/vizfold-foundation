use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .alter_table(
                Table::alter()
                    .table(Runs::Table)
                    .add_column(
                        ColumnDef::new(Runs::InputId)
                            .string()
                            .not_null()
                            .default(""),
                    )
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .alter_table(
                Table::alter()
                    .table(Runs::Table)
                    .drop_column(Runs::InputId)
                    .to_owned(),
            )
            .await
    }
}

#[derive(DeriveIden)]
enum Runs {
    Table,
    InputId,
}
