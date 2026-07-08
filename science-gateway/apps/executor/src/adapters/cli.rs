use crate::core::execution::ExecutionCore;

/// Placeholder for a future CLI adapter that will invoke the shared execution core.
pub struct CliAdapter;

impl CliAdapter {
    #[allow(dead_code)]
    pub async fn run(_core: &ExecutionCore) {
        // CLI commands will be implemented in a later pass.
    }
}
