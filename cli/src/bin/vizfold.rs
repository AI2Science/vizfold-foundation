#[tokio::main]
async fn main() {
    // Not `-> Result<_, DbErr>`: Termination prints an Err with `{:?}`, which wraps every carefully
    // worded message in its enum variant -- `Error: Custom("run 999 does not exist")`. Display is
    // the message itself, modulo the prefix thiserror puts on DbErr::Custom.
    if let Err(error) = executor::adapters::cli::run().await {
        eprintln!("{}", error.to_string().trim_start_matches("Custom Error: "));
        std::process::exit(1);
    }
}
