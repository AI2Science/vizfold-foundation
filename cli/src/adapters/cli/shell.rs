use sea_orm::DbErr;
use std::path::{Path, PathBuf};

pub(super) fn run_to_completion(
    what: &str,
    command: &mut std::process::Command,
) -> Result<(), DbErr> {
    let status = command
        .status()
        .map_err(|error| DbErr::Custom(format!("failed to launch {what}: {error}")))?;
    status
        .success()
        .then_some(())
        // ExitStatus already renders as "exit status: N", so no "exited with status" prefix.
        .ok_or_else(|| DbErr::Custom(format!("{what}: {status}")))
}

/// Executable, not merely present: a failed fetch leaves the truncated file `install.sh` skips over.
pub(super) fn executable(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| {
        meta.is_file() && std::os::unix::fs::PermissionsExt::mode(&meta.permissions()) & 0o111 != 0
    })
}

/// PATH lookup, so nothing has to record where the bootstrap put a core dependency.
pub(super) fn on_path(program: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|dir| dir.join(program))
        .find(|path| executable(path))
}

/// Output, or `None` if unrunnable or killed for running long. Only for output that cannot fill a pipe.
pub(super) fn output_within(
    command: &mut std::process::Command,
    limit: std::time::Duration,
) -> Option<std::process::Output> {
    let mut child = command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let deadline = std::time::Instant::now() + limit;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return child.wait_with_output().ok(),
            Err(_) => return None,
            Ok(None) if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(25)),
        }
    }
}

#[cfg(test)]
mod tests {

    /// A failed fetch leaves a non-executable file; reading it as installed fails deep in an installer.
    #[test]
    fn a_present_but_non_executable_file_is_not_the_dependency() {
        let dir = std::env::temp_dir().join(format!("vizfold-exec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let program = dir.join("micromamba");
        std::fs::write(&program, "").unwrap();

        assert!(!super::executable(&program), "0644 is not runnable");
        std::fs::set_permissions(
            &program,
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )
        .unwrap();
        assert!(super::executable(&program));
        assert!(!super::executable(&dir), "a directory is not a program");
        std::fs::remove_dir_all(&dir).ok();
    }
}
