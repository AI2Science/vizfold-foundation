use super::update::checkout_ref;
use crate::core::examples;
use sea_orm::DbErr;
use std::path::{Path, PathBuf};

use crate::core::{config, release};

use super::args::Backend;
use super::shell::{on_path, output_within};
use super::show::print_table;

/// Runs before any install, so it needs no database.
pub(super) fn run_status() -> Result<(), DbErr> {
    println!("VizFold status\n");
    let components = health();
    print_table(
        &["COMPONENT", "STATUS", "DETAIL"],
        components.iter().map(|component| {
            vec![
                component.name.to_owned(),
                component.state.label().to_owned(),
                component.detail.clone(),
            ]
        }),
    );
    let broken = components.iter().filter(|c| !c.problems.is_empty());
    for (nth, component) in broken.enumerate() {
        println!("{}", if nth == 0 { "\nProblems:" } else { "" });
        for problem in &component.problems {
            println!("  {}: {problem}", component.name);
        }
        println!("  -> {}", component.remedy);
    }
    println!("\n{}\n", summary(&components));

    let config_file = config::config_file();
    if !config::is_initialized() {
        println!("Config: {} (not initialized)", config_file.display());
        return Ok(());
    }
    println!("Config: {}", config_file.display());
    for (key, value) in config::config_entries() {
        // What the checks above read: an inline env var can differ from the file.
        match std::env::var(&key) {
            Ok(inline) if !inline.is_empty() && inline != value => {
                println!("  {key} = {inline}  (env, overriding {value:?})");
            }
            _ => println!("  {key} = {value}"),
        }
    }
    if let Some(database) = config::database_path() {
        let state = if database.is_file() {
            "present"
        } else {
            "not created yet"
        };
        println!("  database = {} ({state})", database.display());
    }
    Ok(())
}

/// One independently breakable part.
#[derive(Default)]
pub(super) struct Component {
    pub(super) name: &'static str,
    pub(super) state: State,
    pub(super) detail: String,
    pub(super) problems: Vec<String>,
    /// Printed under the problems, so it can be set whether or not any were found.
    pub(super) remedy: String,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum State {
    #[default]
    Ok,
    /// Nothing installed it -- most installs run one backend.
    Absent,
    /// Could not be checked from here: no scheduler, no network.
    Unverified,
    Broken,
}

impl State {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Absent => "absent",
            Self::Unverified => "unverified",
            Self::Broken => "BROKEN",
        }
    }
}

/// `Broken` is derived, never tracked: a component that found problems is broken by definition.
pub(super) fn settled(component: Component) -> Component {
    match component.problems.is_empty() {
        true => component,
        false => Component {
            state: State::Broken,
            ..component
        },
    }
}

/// Everything `vizfold status` can settle about an install without folding anything.
pub(super) fn health() -> Vec<Component> {
    [
        core_deps_health(),
        binary_health(),
        repo_health(),
        config_health(),
        backend_health(Backend::Openfold),
        backend_health(Backend::Esmfold),
        scheduler_health(),
    ]
    .into_iter()
    .map(settled)
    .collect()
}

/// The one binary `install.sh` bootstraps; every environment is created and run through it.
pub(super) const CORE_DEP: &str = "micromamba";

pub(super) fn core_deps_health() -> Component {
    let found = on_path(CORE_DEP);
    Component {
        name: "micromamba",
        detail: found
            .as_ref()
            .map_or_else(|| CORE_DEP.to_owned(), |path| path.display().to_string()),
        problems: found
            .is_none()
            .then(|| format!("no executable `{CORE_DEP}` on PATH"))
            .into_iter()
            .collect(),
        remedy: format!(
            "curl -fsSL https://raw.githubusercontent.com/{}/main/install.sh | bash",
            release::repo()
        ),
        ..Default::default()
    }
}

pub(super) fn binary_health() -> Component {
    Component {
        name: "cli",
        detail: release::version_line(release::latest_tag().as_deref()),
        ..Default::default()
    }
}

/// The checkout everything runs from. Drift is flagged only for the clone vizfold made itself.
pub(super) fn repo_health() -> Component {
    repo_health_at(&config::vizfold_repo())
}

pub(super) fn repo_health_at(repo: &Path) -> Component {
    if !repo.join(config::INSTALLER).is_file() {
        return Component {
            name: "repo",
            state: State::Absent,
            detail: format!("no checkout at {}", repo.display()),
            remedy: "vizfold install repo".to_owned(),
            ..Default::default()
        };
    }
    let at = checkout_ref(repo);
    let expected = release::tag();
    let problems = match &at {
        Some(at) if *at != expected && repo == config::default_repo() => {
            vec![format!(
                "the scripts are {at}, but this binary is {expected}"
            )]
        }
        _ => Vec::new(),
    };
    Component {
        name: "repo",
        detail: match &at {
            Some(at) => format!("{} at {at}", repo.display()),
            None => repo.display().to_string(),
        },
        problems,
        remedy: "vizfold update repo".to_owned(),
        ..Default::default()
    }
}

/// Path keys, each with the file proving the claim (empty: the path itself). All must be in `CONFIG_KEYS`.
pub(super) const CHECKED_PATHS: &[(&str, &str)] = &[
    // Not OPENFOLD_HOME: `repo` settles the checkout, and one missing directory must redden one component.
    ("OPENFOLD_PREFIX", ""),
    ("VIZFOLD_ENV_BASE", ""),
];

/// The same, but only while OpenFold is installed: its own uninstall must not read as a broken config.
pub(super) const OPENFOLD_PATHS: &[(&str, &str)] = &[("OPENFOLD_DATA_DIR", "")];

/// Config keys only the scheduler can settle, grouped by the question that answers them.
pub(super) const CHECKED_PARTITIONS: &[&str] = &["OPENFOLD_PARTITION", "OPENFOLD_GPU_PARTITION"];

pub(super) const CHECKED_ACCOUNTS: &[&str] = &["OPENFOLD_ACCOUNT", "OPENFOLD_GPU_ACCOUNT"];

pub(super) fn config_health() -> Component {
    if !config::is_initialized() {
        return Component {
            name: "config",
            state: State::Absent,
            detail: "not initialized".to_owned(),
            remedy: "vizfold install repo".to_owned(),
            ..Default::default()
        };
    }
    let openfold_paths: &[(&str, &str)] = if Backend::Openfold.is_installed() {
        OPENFOLD_PATHS
    } else {
        &[]
    };
    let mut problems: Vec<String> = schema_problem().into_iter().collect();
    problems.extend(
        CHECKED_PATHS
            .iter()
            .chain(openfold_paths)
            .filter_map(|(key, marker)| path_problem(key, marker)),
    );
    Component {
        name: "config",
        detail: format!("{} keys", config::config_keys().len()),
        problems,
        remedy: "vizfold install repo".to_owned(),
        ..Default::default()
    }
}

/// A different key set means a different installer wrote it, so every value under it is suspect.
pub(super) fn schema_problem() -> Option<String> {
    let present = config::config_keys();
    let missing: Vec<&str> = config::CONFIG_KEYS
        .iter()
        .filter(|key| !present.iter().any(|had| had == *key))
        .copied()
        .collect();
    let unknown: Vec<&str> = present
        .iter()
        .filter(|key| !config::CONFIG_KEYS.contains(&key.as_str()))
        .map(String::as_str)
        .collect();
    if missing.is_empty() && unknown.is_empty() {
        return None;
    }
    Some(format!(
        "written by a different vizfold ({})",
        [
            (!missing.is_empty()).then(|| format!("missing {}", missing.join(", "))),
            (!unknown.is_empty()).then(|| format!("unknown {}", unknown.join(", "))),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("; ")
    ))
}

pub(super) fn path_problem(key: &str, marker: &str) -> Option<String> {
    let value = config::resolved(key)?;
    let path = PathBuf::from(&value);
    let proof = if marker.is_empty() {
        path
    } else {
        path.join(marker)
    };
    if proof.exists() {
        return None;
    }
    Some(if marker.is_empty() {
        format!("{key} = {value}, which does not exist")
    } else {
        format!("{key} = {value}, which holds no {marker}")
    })
}

pub(super) fn backend_health(backend: Backend) -> Component {
    let env = backend.env_prefix();
    if !backend.is_installed() {
        return Component {
            name: backend.slug(),
            state: State::Absent,
            detail: format!("not installed ({})", env.display()),
            remedy: format!("vizfold install {}", backend.slug()),
            ..Default::default()
        };
    }
    let mut problems: Vec<String> = missing("interpreter", env.join("bin/python"))
        .into_iter()
        .collect();
    problems.extend(missing("entrypoint", env.join("bin").join(backend.slug())));
    if backend == Backend::Openfold {
        problems.extend(checkout_problem());
        problems.extend(params_problem());
        problems.extend(example_problem());
    }
    Component {
        name: backend.slug(),
        detail: env.display().to_string(),
        problems,
        remedy: format!("vizfold install {}", backend.slug()),
        ..Default::default()
    }
}

/// An editable install into the checkout: losing it breaks every fold at import, env still healthy.
pub(super) fn checkout_problem() -> Option<String> {
    let repo = config::vizfold_repo();
    (!repo.join(config::INSTALLER).is_file()).then(|| {
        format!(
            "the checkout it is installed from is gone: {}",
            repo.display()
        )
    })
}

pub(super) fn missing(what: &str, path: PathBuf) -> Option<String> {
    (!path.is_file()).then(|| format!("no {what} at {}", path.display()))
}

/// Weights resolve under the one data root: `$OPENFOLD_DATA_DIR/params/params_<preset>.npz`, nothing else.
pub(super) fn params_problem() -> Option<String> {
    let weights = config::data_dir().join("params/params_model_1_ptm.npz");
    (!weights.exists()).then(|| {
        format!(
            "AlphaFold2 parameters missing or a dangling link: {}",
            weights.display()
        )
    })
}

/// Half an example -- a FASTA with no alignments -- silently falls back to a full MSA search.
pub(super) fn example_problem() -> Option<String> {
    let id = config::resolved("OPENFOLD_EXAMPLE")?;
    examples::find(&id).is_none().then(|| {
        format!(
            "no FASTA and alignments for OPENFOLD_EXAMPLE {id} under {}",
            examples::monomer_dir().display()
        )
    })
}

/// The difference between a fold that queues and one rejected at submission.
pub(super) fn scheduler_health() -> Component {
    let wanted: Vec<String> = CHECKED_PARTITIONS
        .iter()
        .chain(CHECKED_ACCOUNTS)
        .filter_map(|key| config::resolved(key))
        .collect();
    if wanted.is_empty() {
        return Component {
            name: "scheduler",
            state: State::Absent,
            detail: "no accounts or partitions configured".to_owned(),
            ..Default::default()
        };
    }
    // %100P, not %P: the default width truncates, so a long name reads as a missing partition.
    let partitions = scheduler_values("sinfo", &["-h", "-o", "%100P"]);
    let user = format!("user={}", std::env::var("USER").unwrap_or_default());
    let accounts = scheduler_values(
        "sacctmgr",
        &["-nP", "show", "assoc", &user, "format=Account"],
    );
    if partitions.is_none() && accounts.is_none() {
        return Component {
            name: "scheduler",
            state: State::Unverified,
            detail: format!("{} unchecked: no scheduler here", wanted.join(", ")),
            ..Default::default()
        };
    }
    let problems: Vec<String> = [
        (CHECKED_PARTITIONS, &partitions, "partition"),
        (CHECKED_ACCOUNTS, &accounts, "account"),
    ]
    .into_iter()
    .flat_map(|(keys, known, noun)| {
        keys.iter().filter_map(move |key| {
            unknown_to_scheduler(key, &config::resolved(key)?, known.as_deref(), noun)
        })
    })
    .collect();
    Component {
        name: "scheduler",
        detail: wanted.join(", "),
        problems,
        remedy: "correct them in ~/.config/vizfold/vizfold.json".to_owned(),
        ..Default::default()
    }
}

/// What the scheduler has. `None` means it could not be asked (no command, or slurmctld unreachable).
pub(super) fn scheduler_values(program: &str, args: &[&str]) -> Option<Vec<String>> {
    let output = output_within(
        std::process::Command::new(program).args(args),
        SCHEDULER_TIMEOUT,
    )?;
    if !output.status.success() {
        return None;
    }
    let values = scheduler_names(&String::from_utf8_lossy(&output.stdout));
    (!values.is_empty()).then_some(values)
}

/// Where slurmctld is unreachable, sinfo and sacctmgr block for Slurm's MessageTimeout instead.
pub(super) const SCHEDULER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// One name per line. Trim before stripping `sinfo`'s default-partition `*` -- its padding comes after.
pub(super) fn scheduler_names(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .map(|line| line.trim().trim_end_matches('*').to_owned())
        .filter(|line| !line.is_empty())
        .collect()
}

/// A name the scheduler does not have. `None` known set = unanswered, which is not evidence against it.
pub(super) fn unknown_to_scheduler(
    key: &str,
    value: &str,
    known: Option<&[String]>,
    noun: &str,
) -> Option<String> {
    let known = known?;
    (!known.iter().any(|name| name == value))
        .then(|| format!("{key} names {noun} {value}, which this cluster does not have"))
}

pub(super) fn summary(components: &[Component]) -> String {
    let broken: Vec<&str> = components
        .iter()
        .filter(|component| component.state == State::Broken)
        .map(|component| component.name)
        .collect();
    if broken.is_empty() {
        return "Everything checks out.".to_owned();
    }
    format!(
        "{} of {} components need attention: {}.",
        broken.len(),
        components.len(),
        broken.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Never cloned and drifted need different fixes; both once read BROKEN and pointed at the updater.
    #[test]
    fn an_absent_checkout_is_not_a_broken_one() {
        let _env = crate::core::test_support::env_lock();
        let dir = std::env::temp_dir().join(format!("vizfold-repo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(config::INSTALLER).parent().unwrap()).unwrap();
        let absent = super::repo_health_at(&dir);
        std::fs::write(dir.join(config::INSTALLER), "").unwrap();
        let present = super::repo_health_at(&dir);
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(absent.state, State::Absent);
        assert!(
            absent.problems.is_empty(),
            "absent is not a list of problems"
        );
        assert_eq!(absent.remedy, "vizfold install repo");
        assert_ne!(
            present.state,
            State::Absent,
            "a checkout is there to be judged"
        );
        assert_eq!(present.remedy, "vizfold update repo");
    }

    #[test]
    fn checked_keys_are_all_in_the_schema() {
        let _env = crate::core::test_support::env_lock();
        let checked = super::CHECKED_PATHS
            .iter()
            .chain(super::OPENFOLD_PATHS)
            .map(|(key, _)| *key)
            .chain(super::CHECKED_PARTITIONS.iter().copied())
            .chain(super::CHECKED_ACCOUNTS.iter().copied())
            .chain(["OPENFOLD_EXAMPLE"]);
        for key in checked {
            assert!(
                config::CONFIG_KEYS.contains(&key),
                "{key} is checked but is not a config key"
            );
        }
    }

    /// Real `sinfo -h -o %100P`: a name must survive the `*` marker and the padding, or it reads as missing.
    #[test]
    fn scheduler_names_survive_the_padding_and_the_default_marker() {
        let stdout = format!(
            "{:<100}\n{:<100}\n{:<100}\n\n",
            "cpu*", "cpu*", "gpuA100x4-interactive"
        );
        assert_eq!(
            super::scheduler_names(&stdout),
            ["cpu", "cpu", "gpuA100x4-interactive"]
        );
        assert_eq!(
            super::scheduler_names("bbol-delta-gpu\n"),
            ["bbol-delta-gpu"]
        );
        assert!(super::scheduler_names("\n \n").is_empty());
    }

    #[test]
    fn an_unanswered_scheduler_question_is_not_a_problem() {
        let known = ["cpu".to_owned(), "gpuA100x4-interactive".to_owned()];
        let problem = |known: Option<&[String]>, value| {
            super::unknown_to_scheduler("OPENFOLD_PARTITION", value, known, "partition")
        };
        assert_eq!(problem(None, "cpu"), None, "unasked is not answered");
        assert_eq!(problem(Some(&known), "cpu"), None, "a name it has");
        assert!(
            problem(Some(&known), "gpu-nope")
                .is_some_and(|p| p.contains("gpu-nope") && p.contains("OPENFOLD_PARTITION")),
            "a name it does not have, said with the key that set it"
        );
    }

    /// Only `Broken` counts: an absent backend or unreachable scheduler is not a problem.
    #[test]
    fn the_summary_counts_only_what_is_broken() {
        let component = |name, state| super::Component {
            name,
            state,
            ..Default::default()
        };
        assert_eq!(
            super::summary(&[
                component("cli", super::State::Ok),
                component("esmfold", super::State::Absent),
                component("scheduler", super::State::Unverified),
            ]),
            "Everything checks out."
        );
        assert_eq!(
            super::summary(&[
                component("cli", super::State::Ok),
                component("repo", super::State::Broken),
                component("config", super::State::Broken),
            ]),
            "2 of 3 components need attention: repo, config."
        );
    }

    #[test]
    fn health_promotes_any_component_with_a_problem() {
        for component in super::health() {
            assert_eq!(
                component.state == super::State::Broken,
                !component.problems.is_empty(),
                "{} reports {:?} with {} problem(s)",
                component.name,
                component.state,
                component.problems.len()
            );
        }
    }
}
