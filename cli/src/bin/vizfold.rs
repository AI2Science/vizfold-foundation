#[tokio::main]
async fn main() {
    // Not `-> Result<_, DbErr>`: Termination prints Err with `{:?}`, wrapping the message in its variant.
    if let Err(error) = executor::adapters::cli::run().await {
        eprintln!("{}", error.to_string().trim_start_matches("Custom Error: "));
        std::process::exit(1);
    }
}
