#[tokio::main]
async fn main() -> Result<(), sea_orm::DbErr> {
    executor::adapters::cli::run().await
}
