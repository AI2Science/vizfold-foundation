#[tokio::main]
async fn main() {
    executor::adapters::rest::serve().await;
}
