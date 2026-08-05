use self::args::{InstallArgs, ListArgs, ListResource, ShowResource, UpdateArgs};
use self::install::{run_download, run_install};
use self::run::{default_backend, register_artifacts, run_run};
use self::serve::run_serve;
use self::show::{list_models, list_profiles, list_proteins, list_runs, list_targets, show_run};
use self::status::{
    backend_health, config_health, core_deps_health, repo_health, run_status, settled,
};
use self::uninstall::run_uninstall;
use self::update::{run_self_update, run_update};
use clap::{CommandFactory, Parser};
use clap_complete::Shell;
use sea_orm::DbErr;

use crate::core::db;

use self::args::{Backend, Cli, Command};
use self::status::{Component, State};

mod args;
mod install;
mod run;
mod serve;
mod shell;
mod show;
mod status;
#[cfg(test)]
mod test_support;
mod uninstall;
mod update;

/// The one place a prerequisite is refused, so no command grows its own check.
fn prereqs(command: &Command) -> Vec<Component> {
    // `run <id>` cannot read the run's own backend without the DB, so it gates on the default:
    // uninstalling a backend under a queued run still reaches the runner, which names what is missing.
    let backend = |explicit: Option<Backend>| {
        backend_health(
            explicit
                .or_else(|| default_backend().ok())
                .unwrap_or(Backend::Openfold),
        )
    };
    match command {
        Command::Status
        | Command::Uninstall(_)
        | Command::SelfUpdate(_)
        | Command::Completions(_) => vec![],
        // A backend installs from the checkout; repo's own verbs are what repair it, so they stay off it.
        Command::Install(InstallArgs { part }) | Command::Update(UpdateArgs { part, .. }) => {
            std::iter::once(core_deps_health())
                .chain(part.backend().map(|_| repo_health()))
                .collect()
        }
        Command::List(ListArgs {
            resource: ListResource::Proteins { .. },
        }) => vec![repo_health()],
        // Only the backends named: bare `serve` resolves to what is installed, so it can gate on nothing.
        Command::Serve(args) => [core_deps_health(), repo_health(), config_health()]
            .into_iter()
            .chain(args.backends.iter().copied().map(backend_health))
            .collect(),
        Command::Download(args) => {
            vec![repo_health(), config_health(), backend(Some(args.backend))]
        }
        // Recording a run touches no environment, so a submit host with no micromamba still can.
        Command::Run(args) if args.no_exec => vec![config_health(), backend(args.backend)],
        Command::Run(args) => vec![
            core_deps_health(),
            repo_health(),
            config_health(),
            backend(args.backend),
        ],
        _ => vec![config_health()],
    }
}

/// Ok and Unverified proceed: an unreachable scheduler must not stop a local fold.
fn refuses(state: State) -> bool {
    matches!(state, State::Absent | State::Broken)
}

pub async fn run() -> Result<(), DbErr> {
    let cli = Cli::parse();

    for component in prereqs(&cli.command).into_iter().map(settled) {
        if refuses(component.state) {
            eprintln!("{}: {}", component.name, component.detail);
            for problem in &component.problems {
                eprintln!("  {problem}");
            }
            eprintln!("  -> {}", component.remedy);
            std::process::exit(1);
        }
    }

    match cli.command {
        Command::Install(args) => return run_install(args.part),
        Command::Download(args) => return run_download(args.backend, args.dataset),
        Command::Status => return run_status(),
        Command::Uninstall(args) => return run_uninstall(args),
        Command::Update(args) => return run_update(args),
        Command::SelfUpdate(args) => return run_self_update(args),
        Command::Serve(args) => return run_serve(args),
        Command::Completions(args) => return run_completions(args.shell),
        Command::List(ListArgs {
            resource: ListResource::Proteins { json },
        }) => return list_proteins(json),
        _ => {}
    }

    let database = db::connect_and_migrate().await?;
    match cli.command {
        Command::List(list) => match list.resource {
            ListResource::Models => list_models(&database).await?,
            ListResource::Targets => list_targets(&database).await?,
            ListResource::Profiles => list_profiles(&database).await?,
            ListResource::Runs { status } => list_runs(&database, status.as_deref()).await?,
            ListResource::Proteins { .. } => unreachable!("handled before DB connect"),
        },
        Command::Show(show) => match show.resource {
            ShowResource::Run { run_id } => show_run(&database, run_id).await?,
        },
        Command::Run(args) => run_run(&database, args).await?,
        Command::RegisterArtifacts { run_id } => register_artifacts(&database, run_id).await?,
        Command::Install(_)
        | Command::Download(_)
        | Command::Status
        | Command::Uninstall(_)
        | Command::Update(_)
        | Command::SelfUpdate(_)
        | Command::Serve(_)
        | Command::Completions(_) => {
            unreachable!("handled before DB connect")
        }
    }

    Ok(())
}

/// clap_complete's zsh script registers through `compdef`, which exists only once compinit has run,
/// and a bare ~/.zshrc never runs it. Carried here rather than in each place that evals this.
pub(super) fn completion_script(shell: Shell) -> Vec<u8> {
    let mut script = match shell {
        Shell::Zsh => {
            b"autoload -Uz compinit; (( $+functions[compdef] )) || compinit -C\n".to_vec()
        }
        _ => Vec::new(),
    };
    clap_complete::generate(shell, &mut Cli::command(), "vizfold", &mut script);
    script
}

pub(super) fn run_completions(shell: Option<Shell>) -> Result<(), DbErr> {
    let shell = shell.or_else(Shell::from_env).ok_or_else(|| {
        DbErr::Custom(format!(
            "cannot tell which shell {} is; name one, e.g. `vizfold completions bash`",
            std::env::var("SHELL").unwrap_or_else(|_| "$SHELL".to_owned())
        ))
    })?;
    std::io::Write::write_all(&mut std::io::stdout(), &completion_script(shell))
        .map_err(|error| DbErr::Custom(format!("failed to write completions: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Verified load-bearing: a zsh that never ran compinit registers nothing, and it errors in bash.
    #[test]
    fn only_zsh_carries_the_compinit_prelude() {
        let script = |shell| String::from_utf8(completion_script(shell)).expect("utf-8");
        let zsh = script(Shell::Zsh);
        assert!(
            zsh.starts_with("autoload -Uz compinit;"),
            "the prelude must come before the script that needs compdef, got: {}",
            zsh.lines().next().unwrap_or_default()
        );
        for shell in [Shell::Bash, Shell::Fish] {
            assert!(
                !script(shell).contains("compinit"),
                "{shell} has no compdef to arrange for"
            );
        }
        // Nothing fails visibly if this name is wrong -- completion would simply do nothing.
        assert!(
            zsh.contains("compdef _vizfold vizfold\n"),
            "zsh binds `vizfold`"
        );
        assert!(
            script(Shell::Bash).contains("-F _vizfold -o bashdefault -o default vizfold\n"),
            "bash binds `vizfold`"
        );
    }

    /// The gate table by the names it produces: `status` and repo's own verbs stay off `repo`.
    #[test]
    fn every_command_gates_on_the_prereqs_it_actually_needs() {
        let names = |argv: &[&str]| {
            let cli = Cli::try_parse_from(argv).expect("argv should parse");
            super::prereqs(&cli.command)
                .into_iter()
                .map(|component| component.name)
                .collect::<Vec<_>>()
        };

        assert_eq!(names(&["vizfold", "status"]), Vec::<&str>::new());
        assert_eq!(names(&["vizfold", "uninstall"]), Vec::<&str>::new());
        assert_eq!(names(&["vizfold", "install", "repo"]), ["micromamba"]);
        assert_eq!(names(&["vizfold", "update", "repo"]), ["micromamba"]);
        assert_eq!(
            names(&["vizfold", "install", "openfold"]),
            ["micromamba", "repo"]
        );
        assert_eq!(
            names(&["vizfold", "update", "openfold"]),
            ["micromamba", "repo"]
        );
        assert_eq!(names(&["vizfold", "list", "proteins"]), ["repo"]);
        assert_eq!(
            names(&["vizfold", "serve"]),
            ["micromamba", "repo", "config"]
        );
        assert_eq!(
            names(&["vizfold", "serve", "openfold", "esmfold"]),
            ["micromamba", "repo", "config", "openfold", "esmfold"]
        );
        assert!(Cli::try_parse_from(["vizfold", "serve", "repo"]).is_err());
        assert_eq!(
            names(&["vizfold", "download", "openfold"]),
            ["repo", "config", "openfold"]
        );
        assert_eq!(
            names(&["vizfold", "run", "1UBQ_1"]),
            ["micromamba", "repo", "config", "openfold"]
        );
        // `--no-exec` only writes the row, so it must not gate on an installed environment.
        assert_eq!(
            names(&[
                "vizfold",
                "run",
                "x.fasta",
                "--no-exec",
                "--backend",
                "esmfold"
            ]),
            ["config", "esmfold"]
        );

        let core = super::core_deps_health();
        assert!(
            core.detail.ends_with("micromamba"),
            "core deps must look for micromamba, got {}",
            core.detail
        );
    }

    #[test]
    fn only_absent_and_broken_refuse() {
        assert!(super::refuses(State::Absent));
        assert!(super::refuses(State::Broken));
        assert!(!super::refuses(State::Unverified));
        assert!(!super::refuses(State::Ok));
    }
}
