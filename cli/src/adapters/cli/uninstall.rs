use sea_orm::DbErr;
use std::path::{Path, PathBuf};

use clap::ValueEnum;

use crate::core::config;

use super::args::{Part, UninstallArgs};

/// Undo `vizfold install`. Not a script, because the checkout holding it is one of the things removed.
pub(super) fn run_uninstall(args: UninstallArgs) -> Result<(), DbErr> {
    let (prefix, home) = (config::prefix(), config::openfold_home());
    // One plan for all parts: `removal_plan` folds paths inside the checkout under it, which
    // removing part by part would instead drop as already gone.
    let targets = match args.part {
        Some(part) => part.install_paths(&prefix, &home),
        None => Part::value_variants()
            .iter()
            .flat_map(|part| part.install_paths(&prefix, &home))
            .chain(shared_paths(&prefix, &home))
            .collect(),
    };

    let targets = removal_plan(targets);
    let what = args.part.map_or("vizfold", Part::slug);
    if targets.is_empty() {
        println!("Nothing to remove for {what}.");
        return Ok(());
    }

    let headline = format!("This removes everything {what} installed:");
    if !remove_confirmed(&headline, &targets, args.yes)? {
        return Ok(());
    }
    match args.part {
        Some(_) => println!(
            "\nKept: the config, the run database, and every other part.\nReinstall with: vizfold install {what}"
        ),
        None => {
            // Only once emptied: remove_dir refuses otherwise, which is the whole guard.
            for dir in [
                config::config_file().parent().map(Path::to_path_buf),
                Some(config::env_base()),
                Some(prefix.join("envs")),
            ]
            .into_iter()
            .flatten()
            {
                let _ = std::fs::remove_dir(dir);
            }
            println!(
                "\nKept: fold outputs, and the binaries in ~/.local/bin (vizfold, micromamba)."
            );
        }
    }
    Ok(())
}

/// Print, confirm unless `yes`, remove. `false`: the user declined and nothing went.
pub(super) fn remove_confirmed(
    headline: &str,
    targets: &[PathBuf],
    yes: bool,
) -> Result<bool, DbErr> {
    println!("{headline}");
    for target in targets {
        println!("  {}", target.display());
    }
    if !yes && !confirmed()? {
        println!("Aborted.");
        return Ok(false);
    }
    for target in targets {
        match remove_path(target) {
            Ok(()) => println!("removed {}", target.display()),
            Err(error) => eprintln!("warning: could not remove {}: {error}", target.display()),
        }
    }
    Ok(true)
}

/// A relative path means an empty config value resolved into one -- never delete off the cwd.
pub(super) fn removal_plan(mut targets: Vec<PathBuf>) -> Vec<PathBuf> {
    targets.retain(|path| path.is_absolute() && std::fs::symlink_metadata(path).is_ok());
    targets.sort();
    targets.dedup();
    // Drop what an outer target already covers. ponytail: O(n^2) over ~25 paths.
    let outer = targets.clone();
    targets.retain(|path| {
        !outer
            .iter()
            .any(|other| other != path && path.starts_with(other))
    });
    targets
}

/// What no part owns, so only a full uninstall removes it.
pub(super) fn shared_paths(prefix: &Path, home: &Path) -> Vec<PathBuf> {
    // Named entries, never the env base: only the `vizfold-` ones under it are ours.
    let mut paths = vec![
        config::env_dir("workbench"),
        prefix.join("vizfold.db"),
        config::config_file(),
        // The package cache outlives any one backend.
        prefix.join("mamba"),
    ];
    // The staged copy only; with no prefix settled the two are one path, the repo tree.
    if prefix != home {
        paths.push(prefix.join("workbench"));
    }
    if let Some(database) = config::database_path() {
        let sidecar = |suffix| PathBuf::from(format!("{}{suffix}", database.display()));
        paths.extend([sidecar("-wal"), sidecar("-shm"), database]);
    }
    paths
}

/// `symlink_metadata`, so an AF2 mirror link is removed rather than followed into its directory.
pub(super) fn remove_path(path: &Path) -> std::io::Result<()> {
    if std::fs::symlink_metadata(path)?.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
}

pub(super) fn confirmed() -> Result<bool, DbErr> {
    use std::io::Write;
    print!("Remove these? [y/N] ");
    std::io::stdout().flush().ok();
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|error| DbErr::Custom(format!("could not read confirmation: {error}")))?;
    Ok(matches!(answer.trim(), "y" | "Y" | "yes" | "YES"))
}

#[cfg(test)]
mod tests {
    use super::super::args::repo_paths;
    use super::*;

    /// The reinstall invariant: uninstall takes exactly what install puts back, and nothing else.
    #[test]
    fn one_part_leaves_the_others_and_everything_shared_alone() {
        let base = std::env::temp_dir().join(format!("vizfold-scoped-{}", std::process::id()));
        let (prefix, home) = (base.join("prefix"), base.join("checkout"));

        let owned: Vec<Vec<PathBuf>> = Part::value_variants()
            .iter()
            .map(|part| part.install_paths(&prefix, &home))
            .collect();
        let shared = super::shared_paths(&prefix, &home);

        for (nth, paths) in owned.iter().enumerate() {
            for other in owned.iter().enumerate().filter(|(i, _)| *i != nth) {
                assert!(
                    paths.iter().all(|path| !other.1.contains(path)),
                    "two parts share a path"
                );
            }
        }
        for owned in owned.iter().flatten() {
            assert!(
                !shared.contains(owned),
                "{} is a part's, not shared",
                owned.display()
            );
        }
        assert!(
            repo_paths(&prefix) == vec![config::default_repo()]
                || config::vizfold_repo() != config::default_repo(),
            "repo owns the default checkout and nothing else"
        );
        assert!(
            repo_paths(&config::default_repo().join("prefix")).is_empty(),
            "a prefix inside the checkout keeps the checkout"
        );
        assert!(shared.contains(&config::config_file()), "config is shared");
        // Removing the checkout itself once took a whole home directory with it.
        assert!(
            !shared.contains(&config::env_base()),
            "the env base is the user's; only its vizfold- entries are ours"
        );
        assert!(
            shared.contains(&config::env_dir("workbench")),
            "no backend owns the workbench env, so only a full uninstall takes it"
        );
        assert!(
            shared.contains(&prefix.join("workbench")),
            "a staged workbench is shared"
        );
        assert!(
            !super::shared_paths(&home, &home).contains(&home.join("workbench")),
            "the checkout's own workbench must survive"
        );
    }

    /// `uninstall` is `rm -rf`: no relative path may reach it, and no path an outer target covers.
    #[test]
    fn the_removal_plan_keeps_only_absolute_uncovered_paths_that_exist() {
        let base = std::env::temp_dir().join(format!("vizfold-plan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("outer/inner")).expect("fixture");

        let plan = super::removal_plan(vec![
            // Exists relative to the cwd tests run in: without the guard, uninstall deletes it.
            PathBuf::from("Cargo.toml"),
            base.join("does-not-exist"),
            base.join("outer/inner"),
            base.join("outer"),
            base.join("outer"), // duplicate: dedup is what keeps the plan one entry
        ]);

        assert_eq!(plan, vec![base.join("outer")]);
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn remove_path_unlinks_a_symlink_without_touching_its_target() {
        let base = std::env::temp_dir().join(format!("vizfold-rm-{}", std::process::id()));
        let (mirror, link) = (base.join("mirror"), base.join("params"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&mirror).unwrap();
        std::fs::write(mirror.join("params.npz"), "keep").unwrap();
        std::os::unix::fs::symlink(&mirror, &link).unwrap();

        super::remove_path(&link).unwrap();

        assert!(link.symlink_metadata().is_err(), "symlink should be gone");
        assert!(mirror.join("params.npz").is_file(), "target must survive");
        super::remove_path(&mirror).unwrap();
        assert!(!mirror.exists(), "a real directory is removed whole");
        std::fs::remove_dir_all(&base).ok();
    }
}
