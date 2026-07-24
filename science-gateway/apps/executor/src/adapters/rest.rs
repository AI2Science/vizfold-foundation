use axum::{Json, Router, routing::get};
use serde_json::json;
use tokio::net::TcpListener;

use crate::core::{config, db::ExecutionCore};

pub async fn serve() {
    let _ = dotenvy::dotenv();

    let core = ExecutionCore::bootstrap()
        .await
        .expect("failed to initialize execution core");

    core.check_readiness()
        .await
        .expect("failed to query model backends");

    let app = Router::new().route("/health", get(health));
    let listener = TcpListener::bind("127.0.0.1:3001")
        .await
        .expect("failed to bind TCP listener");

    println!("API listening on http://127.0.0.1:3001");
    println!("Database connected using {}", config::database_url());
    let _ = core.db();

    axum::serve(listener, app).await.expect("server error");
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}
