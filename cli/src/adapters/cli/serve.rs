use sea_orm::DbErr;
use std::path::{Path, PathBuf};

use crate::core::config;

use super::args::ServeArgs;
use super::shell::run_to_completion;

/// Stage the dashboard, provision Bun, and install its dependencies. Idempotent, and the reason
/// `serve` still calls it: an install that predates the dashboard, or a cleared prefix, self-heals.
pub(super) fn ensure_dashboard(workbench: &Path, backends: &str) -> Result<PathBuf, DbErr> {
    let bun_bin = ensure_bun()?;
    let node_modules = workbench.join("node_modules");
    let empty =
        std::fs::read_dir(&node_modules).map_or(true, |mut entries| entries.next().is_none());
    if empty {
        println!("Installing workbench dependencies (bun install)...");
        run_bun(workbench, &bun_bin, &["install"], backends, None)?;
    }
    Ok(bun_bin)
}

pub(super) fn run_serve(args: ServeArgs) -> Result<(), DbErr> {
    let workbench = serve_dir()?;

    // The dashboard serves each run's files out of the run directory itself, checked against it,
    // so there is nothing to link into a public/ directory.
    // ponytail: seeded output_location only; read it from provenance if a profile ever differs.
    std::fs::create_dir_all(config::prefix().join("runs")).ok();

    let backends = args.backends_env();
    let bun_bin = ensure_dashboard(&workbench, &backends)?;

    let port = args.port.unwrap_or(3000);
    println!(
        "Starting workbench at http://localhost:{port} ({})",
        if backends.is_empty() {
            "no backend installed"
        } else {
            &backends
        }
    );
    run_bun(
        &workbench,
        &bun_bin,
        &["run", "start"],
        &backends,
        Some(port),
    )
}

/// Where the dashboard runs from: a copy on the prefix's filesystem, so node_modules never lands on home.
pub(super) fn serve_dir() -> Result<PathBuf, DbErr> {
    let repo = config::openfold_home().join("workbench");
    if config::prefix() == config::openfold_home() {
        return Ok(repo);
    }
    let dest = config::prefix().join("workbench");
    copy_tree(&repo, &dest, &["node_modules", "dist"]).map_err(|error| {
        DbErr::Custom(format!(
            "failed to stage workbench at '{}': {error}",
            dest.display()
        ))
    })?;
    Ok(dest)
}

/// Merge `repo` into `dst`, skipping the named top-level entries (build output, neither copied nor clobbered).
pub(super) fn copy_tree(repo: &Path, dst: &Path, skip: &[&str]) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(repo)? {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_str().is_some_and(|n| skip.contains(&n)) {
            continue;
        }
        let file_type = entry.file_type()?;
        // fs::copy follows a symlink, and a link-to-directory hits EISDIR. Recreated at serve time.
        if file_type.is_symlink() {
            continue;
        }
        let (from, to) = (entry.path(), dst.join(&name));
        if file_type.is_dir() {
            copy_tree(&from, &to, &[])?; // skip applies only at the workbench root
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// The workbench is a Bun program: `bun:sqlite`, `Bun.serve` and its bundler, no Node in the loop.
/// 1.2 is where the HTML-entry server the dashboard is written against landed.
pub(super) const MIN_BUN: (u32, u32) = (1, 2);

/// Unparseable reads as too old: provisioning a known-good Bun beats a failure deep in the server.
pub(super) fn bun_is_new_enough(version: &str) -> bool {
    let mut parts = version.trim().trim_start_matches('v').split('.');
    let (Some(Ok(major)), Some(Ok(minor))) = (
        parts.next().map(str::parse::<u32>),
        parts.next().map(str::parse::<u32>),
    ) else {
        return false;
    };
    (major, minor) >= MIN_BUN
}

/// The bun on PATH, when it is new enough. Its own directory, so the server's child processes
/// resolve the same one.
pub(super) fn system_bun_bin() -> Option<PathBuf> {
    let version = std::process::Command::new("bun")
        .arg("--version")
        .output()
        .ok()?;
    if !bun_is_new_enough(&String::from_utf8_lossy(&version.stdout)) {
        return None;
    }
    let which = std::process::Command::new("bun")
        .args(["--eval", "process.stdout.write(process.execPath)"])
        .output()
        .ok()?;
    let path = PathBuf::from(String::from_utf8(which.stdout).ok()?);
    path.parent().map(Path::to_path_buf)
}

/// The release asset for this machine. The musl builds are statically linked, so they run on a
/// cluster whose glibc predates the gnu build's floor.
pub(super) fn bun_asset() -> Result<&'static str, DbErr> {
    match std::env::consts::ARCH {
        "x86_64" => Ok("bun-linux-x64-musl"),
        "aarch64" => Ok("bun-linux-aarch64-musl"),
        other => Err(DbErr::Custom(format!(
            "no Bun release for {other}; install bun >= {}.{} yourself and re-run",
            MIN_BUN.0, MIN_BUN.1
        ))),
    }
}

/// Bun from PATH when new enough, else one downloaded beside the workbench -- kept out of every
/// backend env, and off `$HOME`, which is the quota'd filesystem this staging exists to avoid.
pub(super) fn ensure_bun() -> Result<PathBuf, DbErr> {
    let bin = config::env_dir("workbench").join("bin");
    if bin.join("bun").is_file() {
        return Ok(bin);
    }
    if let Some(system) = system_bun_bin() {
        return Ok(system);
    }

    let asset = bun_asset()?;
    println!("Downloading Bun (first run only)...");
    std::fs::create_dir_all(&bin)
        .map_err(|error| DbErr::Custom(format!("failed to create '{}': {error}", bin.display())))?;
    let archive = bin.join("bun.zip");
    run_to_completion(
        "downloading Bun",
        std::process::Command::new("curl")
            .args(["-fsSL", "-o"])
            .arg(&archive)
            .arg(format!(
                "https://github.com/oven-sh/bun/releases/latest/download/{asset}.zip"
            )),
    )?;
    unzip(&archive, &bin)?;
    // The archive holds `<asset>/bun`; lift it out and drop the rest.
    std::fs::rename(bin.join(asset).join("bun"), bin.join("bun"))
        .map_err(|error| DbErr::Custom(format!("Bun archive did not hold {asset}/bun: {error}")))?;
    std::fs::remove_dir_all(bin.join(asset)).ok();
    std::fs::remove_file(&archive).ok();
    std::fs::set_permissions(
        bin.join("bun"),
        std::os::unix::fs::PermissionsExt::from_mode(0o755),
    )
    .map_err(|error| DbErr::Custom(format!("failed to make bun executable: {error}")))?;
    Ok(bin)
}

/// `unzip` where the machine has it, else Python's zipfile module -- a login node has one or the
/// other, and Bun publishes zips only.
pub(super) fn unzip(archive: &Path, into: &Path) -> Result<(), DbErr> {
    let unzip = std::process::Command::new("unzip")
        .arg("-oq")
        .arg(archive)
        .arg("-d")
        .arg(into)
        .status();
    if matches!(&unzip, Ok(status) if status.success()) {
        return Ok(());
    }
    run_to_completion(
        "extracting Bun",
        std::process::Command::new("python3")
            .args(["-m", "zipfile", "-e"])
            .arg(archive)
            .arg(into),
    )
    .map_err(|error| {
        DbErr::Custom(format!(
            "{error}; neither `unzip` nor `python3` could extract '{}'",
            archive.display()
        ))
    })
}

/// Run bun in the staged workbench, with everything the dashboard reads off `process.env`.
pub(super) fn run_bun(
    dir: &Path,
    bun_bin: &Path,
    args: &[&str],
    backends: &str,
    port: Option<u16>,
) -> Result<(), DbErr> {
    let mut command = std::process::Command::new(bun_bin.join("bun"));
    command.current_dir(dir).args(args);
    // `bun run start` re-invokes bun for the script, and resolves it off PATH.
    let path = std::env::var_os("PATH").unwrap_or_default();
    let dirs = std::iter::once(bun_bin.to_path_buf()).chain(std::env::split_paths(&path));
    command.env(
        "PATH",
        std::env::join_paths(dirs)
            .map_err(|error| DbErr::Custom(format!("failed to build PATH: {error}")))?,
    );
    // The dashboard shells out to this binary, and logs each background fold under the prefix.
    if let Ok(binary) = std::env::current_exe() {
        command.env("VIZFOLD_BIN", binary);
    }
    // Bun caches in $HOME/.bun by default -- inodes on the quota'd home, again.
    command.env(
        "BUN_INSTALL_CACHE_DIR",
        config::env_dir("workbench").join(".bun-cache"),
    );
    command.env("OPENFOLD_PREFIX", config::prefix());
    // The resolved set, always explicit, so the dashboard needs no notion of what "all" means.
    command.env("VIZFOLD_BACKENDS", backends);
    // bun:sqlite cannot open database_url()'s sqlite://...?mode=rwc wrapper; hand it the plain path.
    if let Some(database) = config::database_path() {
        command.env("VIZFOLD_DB", database);
    }
    if let Some(port) = port {
        command.env("PORT", port.to_string());
    }
    run_to_completion(&format!("bun {}", args.join(" ")), &mut command)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_tree_excludes_build_artifacts_and_preserves_dest() {
        let base = std::env::temp_dir().join(format!("vizfold-copytree-{}", std::process::id()));
        let (repo, dst) = (base.join("repo"), base.join("dst"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(repo.join("node_modules")).unwrap();
        std::fs::create_dir_all(repo.join("dist")).unwrap();
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::write(repo.join("package.json"), "{}").unwrap();
        std::fs::write(repo.join("node_modules/dep.js"), "repo").unwrap();
        std::fs::write(repo.join("src/main.tsx"), "x").unwrap();
        // A node_modules already staged in the destination must survive the copy.
        std::fs::create_dir_all(dst.join("node_modules")).unwrap();
        std::fs::write(dst.join("node_modules/installed.js"), "keep").unwrap();

        super::copy_tree(&repo, &dst, &["node_modules", "dist"]).unwrap();

        assert!(dst.join("package.json").is_file());
        assert!(dst.join("src/main.tsx").is_file());
        assert!(!dst.join("dist").exists());
        assert!(dst.join("node_modules/installed.js").is_file());
        assert!(!dst.join("node_modules/dep.js").exists());
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn copy_tree_skips_a_symlinked_directory() {
        let base =
            std::env::temp_dir().join(format!("vizfold-copytree-link-{}", std::process::id()));
        let (repo, dst, outputs) = (base.join("repo"), base.join("dst"), base.join("outputs"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(repo.join("public")).unwrap();
        std::fs::create_dir_all(&outputs).unwrap();
        std::fs::write(repo.join("package.json"), "{}").unwrap();
        std::os::unix::fs::symlink(&outputs, repo.join("public/runs")).unwrap();

        super::copy_tree(&repo, &dst, &[]).unwrap();

        assert!(dst.join("package.json").is_file());
        assert!(!dst.join("public/runs").exists());
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn bun_version_gate_is_major_then_minor() {
        assert!(bun_is_new_enough("1.2.0"));
        assert!(
            bun_is_new_enough("v1.3.11\n"),
            "`bun --version` ends in a newline"
        );
        assert!(bun_is_new_enough("2.0.1"));
        assert!(!bun_is_new_enough("1.1.45"), "minor below the floor");
        assert!(!bun_is_new_enough("0.8.1"), "major below the floor");
        // 1.10 must beat 1.2 -- the reason this compares numbers, not strings.
        assert!(bun_is_new_enough("1.10.0"));
        assert!(!bun_is_new_enough(""), "unparseable reads as too old");
        assert!(!bun_is_new_enough("garbage"));
    }

    #[test]
    fn bun_asset_is_the_static_build_for_this_machine() {
        // Only the two architectures the releases are published for; both musl, so a cluster whose
        // glibc predates the gnu build still runs the dashboard.
        assert!(matches!(
            super::bun_asset(),
            Ok("bun-linux-x64-musl") | Ok("bun-linux-aarch64-musl")
        ));
    }
}
