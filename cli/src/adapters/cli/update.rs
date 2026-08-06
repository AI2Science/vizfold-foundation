use super::uninstall::removal_plan;
use sea_orm::DbErr;
use std::path::{Path, PathBuf};

use crate::core::{config, release};

use super::args::{Backend, SelfUpdateArgs, UpdateArgs};
use super::install::install_backend;
use super::shell::run_to_completion;
use super::uninstall::remove_confirmed;

pub(super) fn run_update(args: UpdateArgs) -> Result<(), DbErr> {
    match args.part.backend() {
        None => update_repo(args.r#ref.as_deref()),
        Some(backend) if args.r#ref.is_some() => Err(DbErr::Custom(format!(
            "--ref moves the checkout, so it belongs to `vizfold update repo`, not {}",
            backend.slug()
        ))),
        Some(backend) => reinstall(backend, args.yes),
    }
}

/// Neither installer reruns on drift, so a fresh checkout's scripts reach the env only through this.
pub(super) fn reinstall(backend: Backend, yes: bool) -> Result<(), DbErr> {
    let targets = removal_plan(reinstall_paths(
        backend,
        &config::prefix(),
        &config::openfold_home(),
        &config::data_dir(),
    ));
    let headline = format!("Reinstalling {} first removes:", backend.slug());
    if !targets.is_empty() && !remove_confirmed(&headline, &targets, yes)? {
        return Ok(());
    }
    install_backend(backend)
}

/// Everything install planted except downloads: params are ~4 GB, or a mirror symlink tree, and
/// neither is install state. `data` may sit under a planted directory, spent entry by entry instead.
pub(super) fn reinstall_paths(
    backend: Backend,
    prefix: &Path,
    home: &Path,
    data: &Path,
) -> Vec<PathBuf> {
    backend
        .install_paths(prefix, home)
        .into_iter()
        .flat_map(|path| {
            if !data.starts_with(&path) {
                return vec![path];
            }
            std::fs::read_dir(&path)
                .into_iter()
                .flatten()
                .flatten()
                .map(|entry| entry.path())
                .filter(|entry| !data.starts_with(entry))
                .collect()
        })
        .collect()
}

/// Move the checkout to a ref, by default this binary's own release tag.
pub(super) fn update_repo(wanted: Option<&str>) -> Result<(), DbErr> {
    let repo = config::vizfold_repo();
    let target = wanted.unwrap_or(&release::tag()).to_owned();
    if !repo.join(config::INSTALLER).is_file() {
        return Err(DbErr::Custom(format!(
            "no vizfold checkout at {}; create one with `vizfold install repo`",
            repo.display()
        )));
    }
    if !repo.join(".git").exists() {
        return Err(DbErr::Custom(format!(
            "{} is not a git checkout; nothing to update",
            repo.display()
        )));
    }
    // Tracked edits only: the install builds OpenFold's CUDA extension in this checkout.
    match git(&repo, &["status", "--porcelain", "--untracked-files=no"]) {
        None => {
            return Err(DbErr::Custom(format!(
                "cannot read `git status` in {}; is git on PATH and the checkout yours to read?",
                repo.display()
            )));
        }
        Some(changes) if !changes.trim().is_empty() => {
            return Err(DbErr::Custom(format!(
                "{} has uncommitted changes; commit or discard them first",
                repo.display()
            )));
        }
        _ => {}
    }
    println!("Updating {} to {target} ...", repo.display());
    // Shallow single-branch clone: the ref must be fetched by name, and only FETCH_HEAD names it after.
    run_to_completion(
        "fetch",
        &mut git_cmd(
            &repo,
            &["fetch", "--depth", "1", "--tags", "origin", &target],
        ),
    )?;
    run_to_completion(
        "checkout",
        &mut git_cmd(&repo, &["checkout", "--force", "FETCH_HEAD"]),
    )?;
    println!(
        "{} is now at {}",
        repo.display(),
        checkout_ref(&repo).unwrap_or(target)
    );
    Ok(())
}

pub(super) fn git_cmd(dir: &Path, args: &[&str]) -> std::process::Command {
    let mut command = std::process::Command::new("git");
    command.arg("-C").arg(dir).args(args);
    command
}

/// One read-only git question as trimmed stdout; `None` when git cannot answer.
pub(super) fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let output = git_cmd(dir, args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

pub(super) fn checkout_ref(repo: &Path) -> Option<String> {
    git(repo, &["describe", "--tags", "--exact-match"])
        .or_else(|| git(repo, &["rev-parse", "--short", "HEAD"]))
        .filter(|value| !value.is_empty())
}

/// Replace this binary, then let the new one update its own checkout. Staged beside it: rename is per-fs.
pub(super) fn run_self_update(args: SelfUpdateArgs) -> Result<(), DbErr> {
    let exe = std::env::current_exe()
        .map_err(|error| DbErr::Custom(format!("cannot locate this binary: {error}")))?;
    let current = release::current();
    let tag = match args.version {
        Some(version) => version,
        None => release::latest_tag().ok_or_else(|| {
            DbErr::Custom(
                "could not reach github.com for the latest release; pass --version to pin one"
                    .to_owned(),
            )
        })?,
    };
    let wanted = release::version_of(&tag).to_owned();
    if wanted == current && !args.force {
        println!("vizfold {current} is already the latest release.");
        return Ok(());
    }
    let asset = release::asset(std::env::consts::ARCH);
    let url = release::asset_url(&tag, &asset);
    let staged = exe.with_file_name(format!(".{asset}.incoming"));
    println!("Updating vizfold {current} -> {wanted}\n  {url}");
    let fetched = fetch_release(&url, &staged, &wanted);
    if fetched.is_err() {
        let _ = std::fs::remove_file(&staged);
    }
    fetched?;
    std::fs::rename(&staged, &exe).map_err(|error| {
        let _ = std::fs::remove_file(&staged);
        DbErr::Custom(format!(
            "cannot replace {}: {error}; re-run where you can write it",
            exe.display()
        ))
    })?;
    println!("vizfold {wanted} is installed at {}", exe.display());
    println!(
        "The checkout still runs {current}'s scripts. Bring it along with: vizfold update repo"
    );
    Ok(())
}

/// Prove the download is a working binary of the version it claims before it replaces anything.
pub(super) fn fetch_release(url: &str, staged: &Path, wanted: &str) -> Result<(), DbErr> {
    run_to_completion(
        "download",
        std::process::Command::new("curl")
            .args(["-fsSL", url, "-o"])
            .arg(staged),
    )?;
    std::fs::set_permissions(staged, std::os::unix::fs::PermissionsExt::from_mode(0o755))
        .map_err(|error| DbErr::Custom(format!("cannot make the download executable: {error}")))?;
    let reported = std::process::Command::new(staged)
        .arg("--version")
        .output()
        .map_err(|error| DbErr::Custom(format!("the download will not run here: {error}")))?;
    let reported = String::from_utf8_lossy(&reported.stdout).trim().to_owned();
    if !reported.contains(wanted) {
        return Err(DbErr::Custom(format!(
            "the download reports itself as {reported:?}, not {wanted}; leaving this binary alone"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    use super::super::args::{Cli, Command};

    #[test]
    fn a_reinstall_keeps_the_downloads_and_takes_everything_else() {
        let _env = crate::core::test_support::env_lock();
        let base = std::env::temp_dir().join(format!("vizfold-reinstall-{}", std::process::id()));
        let (prefix, home) = (base.join("prefix"), base.join("checkout"));
        let full = Backend::Openfold.install_paths(&prefix, &home);
        let kept = super::reinstall_paths(Backend::Openfold, &prefix, &home, &config::data_dir());

        assert!(
            full.contains(&config::data_dir()),
            "install writes the data dir, or there is nothing to exclude"
        );
        assert!(
            !kept.contains(&config::data_dir()),
            "downloads are not install state"
        );
        assert!(
            kept.contains(&Backend::Openfold.env_prefix()),
            "the env is rebuilt"
        );
        assert_eq!(kept.len(), full.len() - 1, "only the data dir is spared");
    }

    /// The data dir sits inside the state dir install also plants: removing that whole takes the databases.
    #[test]
    fn a_reinstall_removes_no_directory_the_downloads_live_under() {
        let _env = crate::core::test_support::env_lock();
        let base = std::env::temp_dir().join(format!("vizfold-nested-data-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (prefix, home) = (base.join("prefix"), base.join("checkout"));
        let state = prefix.join("openfold");
        let data = state.join("data");
        std::fs::create_dir_all(&data).expect("fixture");
        std::fs::create_dir_all(state.join("pkgs")).expect("fixture");
        std::fs::write(state.join(".done"), "").expect("fixture");

        let kept = super::reinstall_paths(Backend::Openfold, &prefix, &home, &data);
        std::fs::remove_dir_all(&base).ok();

        assert!(
            !kept.iter().any(|path| data.starts_with(path)),
            "a removal target holds the downloads: {kept:?}"
        );
        // Without these the installer short-circuits on the sentinel.
        assert!(kept.contains(&state.join(".done")), "the sentinel must go");
        assert!(
            kept.contains(&state.join("pkgs")),
            "the rest of the state dir must go"
        );
    }

    /// Only repo takes a ref; silently ignoring it on a backend would fake a version move.
    #[test]
    fn a_ref_on_a_backend_update_is_refused() {
        let update = |argv: &[&str]| match Cli::try_parse_from(argv).expect("argv").command {
            Command::Update(args) => args,
            other => panic!("not an update: {other:?}"),
        };
        assert!(
            super::run_update(update(&["vizfold", "update", "esmfold", "--ref", "v0.1.0"]))
                .is_err_and(|error| format!("{error}").contains("vizfold update repo"))
        );
    }
}
