mod adapters;
mod core;

#[tokio::main]
async fn main() {
    adapters::rest::serve().await;
}
