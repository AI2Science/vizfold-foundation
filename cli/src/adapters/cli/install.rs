use sea_orm::DbErr;

use crate::core::{config, release};

use super::args::{Backend, Part};
use super::serve::{ensure_dashboard, serve_dir};
use super::shell::run_to_completion;

/// Make a part exist. Idempotent: both backend installers short-circuit on presence, as repo does.
pub(super) fn run_install(part: Part) -> Result<(), DbErr> {
    match part.backend() {
        Some(backend) => install_backend(backend),
        None => install_repo(),
    }
}

/// Fetch the checkout the binary installs everything from -- it ships only itself. Never moves one.
/// The checkout, then the config that everything downstream reads: which cluster, which prefix,
/// which AlphaFold2 mirror holds the protein databases, what the scheduler takes. Re-runnable --
/// `config::load` restores the previous answers, so this settles what is missing and keeps the rest.
pub(super) fn install_repo() -> Result<(), DbErr> {
    let repo = config::vizfold_repo();
    if repo.join(config::INSTALLER).is_file() {
        println!("{} already is a vizfold checkout.", repo.display());
    } else {
        clone_checkout(&repo)?;
    }
    run_to_completion(
        "configure",
        std::process::Command::new("bash")
            .arg(repo.join(CONFIGURE))
            .env("OPENFOLD_HOME", &repo),
    )?;
    // configure.sh just wrote the prefix; without this the dashboard would be staged under the
    // default this process started with, which is a directory the install never settled on.
    config::reload();
    // The dashboard too, so `serve` starts rather than provisioning: on a cluster that is a Bun
    // download and a `bun install`, and `serve` is run when someone wants to look.
    ensure_dashboard(&serve_dir()?, "")?;
    Ok(())
}

/// Settles the site and writes the config; `install repo` owns both.
pub(super) const CONFIGURE: &str = "configure.sh";

pub(super) fn install_backend(backend: Backend) -> Result<(), DbErr> {
    let repo = config::vizfold_repo();
    let installer = repo.join(backend.installer());
    println!(
        "Installing {}: bash {}",
        backend.slug(),
        installer.display()
    );
    run_to_completion(
        "model install",
        std::process::Command::new("bash")
            .arg(&installer)
            .env("OPENFOLD_HOME", &repo),
    )
}

pub(super) fn run_download(backend: Backend, dataset: String) -> Result<(), DbErr> {
    let Some(dir) = backend.downloader_dir() else {
        println!(
            "{} fetches its weights from HuggingFace at run time; nothing to download.",
            backend.slug()
        );
        return Ok(());
    };
    let repo = config::vizfold_repo();
    let script_name = if dataset == "all" {
        "download_alphafold_dbs.sh".to_string()
    } else {
        format!("download_{dataset}.sh")
    };
    let script = repo.join(dir).join(&script_name);
    if !script.is_file() {
        return Err(DbErr::Custom(format!(
            "no downloader `{script_name}` at {}; pass `all` or a db name (e.g. uniref90, pdb70, bfd)",
            script.display()
        )));
    }
    let dest = config::data_dir();
    std::fs::create_dir_all(&dest)
        .map_err(|error| DbErr::Custom(format!("cannot create {}: {error}", dest.display())))?;
    println!(
        "Downloading {} `{dataset}`: bash {} {}",
        backend.slug(),
        script.display(),
        dest.display()
    );
    // The downloaders need aria2c and aws, which live only inside the OpenFold environment.
    let path = std::env::var("PATH").unwrap_or_default();
    let env_bin = config::openfold_env_prefix().join("bin");
    run_to_completion(
        "downloader",
        std::process::Command::new("bash")
            .arg(&script)
            .arg(&dest)
            .env("OPENFOLD_HOME", &repo)
            .env("PATH", format!("{}:{path}", env_bin.display())),
    )
}

/// Clone the checkout, pinned to `release::tag()` -- the default branch when no such tag exists.
pub(super) fn clone_checkout(repo: &std::path::Path) -> Result<(), DbErr> {
    let url = format!("https://github.com/{}.git", release::repo());
    let dest = repo.to_string_lossy().into_owned();
    println!("Fetching the vizfold checkout into {dest} ...");
    let clone = |args: &[&str]| std::process::Command::new("git").args(args).status();
    if let Ok(status) = clone(&[
        "clone",
        "--depth",
        "1",
        "--branch",
        &release::tag(),
        &url,
        &dest,
    ]) && status.success()
    {
        return Ok(());
    }
    match clone(&["clone", "--depth", "1", &url, &dest]) {
        Ok(status) if status.success() => Ok(()),
        _ => Err(DbErr::Custom(format!(
            "failed to clone {url} into {dest}; set OPENFOLD_HOME to an existing checkout"
        ))),
    }
}
