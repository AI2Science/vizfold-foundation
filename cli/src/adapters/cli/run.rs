use crate::core::seed::seed_defaults;
use sea_orm::{ColumnTrait, DbErr, EntityTrait, QueryFilter};
use serde_json::json;
use std::path::{Path, PathBuf};

use crate::core::{
    commands::LocalCommandRunner,
    config,
    entities::{
        execution_targets as execution_target_entity, model_backends as model_backend_entity,
    },
    examples,
    output_locations::resolve_output_location,
    preflight::PreflightStatus,
    services::{model_invocation_profiles, run_artifacts, run_execution::execute_run, runs},
};

use super::args::{Backend, RunArgs};
use super::show::print_table;

/// A GPU partition with no allocation held means the fold is srun'd onto a GPU node regardless of this host.
pub(super) fn on_gpu_partition(context: config::SlurmContext, partition: Option<&str>) -> bool {
    matches!(context, config::SlurmContext::None) && partition.is_some_and(|p| !p.is_empty())
}

pub(super) fn model_device_for(
    context: config::SlurmContext,
    partition: Option<&str>,
    detected: Option<&str>,
) -> String {
    if on_gpu_partition(context, partition) || detected.is_some() {
        "cuda:0".to_owned()
    } else {
        "cpu".to_owned()
    }
}

/// Skips the local probe when it would not be consulted, rather than discarding its result.
pub(super) fn default_model_device() -> String {
    let context = config::SlurmContext::detect();
    let partition = config::gpu_partition();
    let detected = if on_gpu_partition(context, partition.as_deref()) {
        None
    } else {
        crate::core::preflight::detect_gpu()
    };
    model_device_for(context, partition.as_deref(), detected.as_deref())
}

pub(super) fn default_cpus() -> i64 {
    std::thread::available_parallelism().map_or(1, |n| n.get() as i64)
}

/// Clamp to the target's `cpus.maximum`, so a host with more cores still queues a runnable plan.
pub(super) fn clamp_cpus(cpus: i64, available_resources_json: &str) -> i64 {
    let max_cpus = serde_json::from_str::<serde_json::Value>(available_resources_json)
        .ok()
        .and_then(|resources| resources["properties"]["cpus"]["maximum"].as_i64())
        .unwrap_or(i64::MAX);
    cpus.min(max_cpus)
}

pub(super) async fn register_artifacts(
    database: &sea_orm::DatabaseConnection,
    run_id: i32,
) -> Result<(), DbErr> {
    let run = runs::get_run_with_artifacts(database, run_id)
        .await?
        .ok_or_else(|| DbErr::Custom(format!("run {run_id} does not exist")))?
        .run;
    if run.status != "completed" {
        println!(
            "Warning: run {run_id} has status '{}'; registered artifacts may be partial.",
            run.status
        );
    }
    let backend = model_backend_entity::Entity::find_by_id(run.model_backend_id)
        .one(database)
        .await?
        .ok_or_else(|| DbErr::Custom("run model backend does not exist".into()))?;
    if !matches!(backend.slug.as_str(), "openfold" | "esmfold") {
        return Err(DbErr::Custom(format!(
            "artifact registration is currently only implemented for OpenFold and ESMFold runs (run {run_id} uses backend '{}')",
            backend.slug
        )));
    }

    let profile =
        crate::core::services::require_profile(database, run.invocation_profile_id).await?;
    let workspace = resolve_output_location(&profile, &run)?;
    let added = run_artifacts::register_run_artifacts(database, run_id).await?;
    let held = run_artifacts::artifact_counts_by_kind(database, run_id).await?;

    println!("Registered artifacts for run {run_id}");
    println!("\nOutput workspace:\n  {}", workspace.display());
    if held.is_empty() {
        println!("\nNothing written yet: the run produced no files to register.");
        return Ok(());
    }

    // What each kind holds now, and how much of it this pass is responsible for -- a re-register
    // over an unchanged workspace is all zeros, which is the answer to "did anything land".
    println!("\nArtifacts by kind:");
    print_table(
        &["KIND", "HELD", "NEW"],
        held.iter().map(|(slug, count)| {
            vec![
                slug.clone(),
                count.to_string(),
                added
                    .get(slug.as_str())
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
            ]
        }),
    );
    Ok(())
}

/// The execution alone; `run_run` owns the queueing and registration around it.
pub(super) async fn report_execution(
    database: &sea_orm::DatabaseConnection,
    run_id: i32,
) -> Result<(), DbErr> {
    println!("Executing run {run_id}");
    let outcome = execute_run(database, run_id, &LocalCommandRunner).await?;

    let label = if outcome.report.has_failures() {
        "failed"
    } else {
        "passed"
    };
    println!("\nPreflight: {label}");
    for check in outcome.report.checks {
        let message = check.message.as_deref().unwrap_or("no details");
        println!(
            "[{}] {}: {}",
            preflight_status_label(check.status),
            check.name,
            message
        );
    }

    // Only exit_code: it streamed, so stdout/stderr are empty by construction.
    if let Some(output) = outcome.output {
        println!("\nCommand exit_code: {}", output.exit_code);
    }

    if let Some(run) = runs::get_run_with_artifacts(database, run_id)
        .await?
        .map(|result| result.run)
    {
        println!("\nFinal status: {}", run.status);
    }
    Ok(())
}

pub(super) fn preflight_status_label(status: PreflightStatus) -> &'static str {
    match status {
        PreflightStatus::Passed => "passed",
        PreflightStatus::Warning => "warning",
        PreflightStatus::Failed => "failed",
    }
}

/// Fold every target in one execution, or replay a queued run by id. Re-registers artifacts (idempotent).
pub(super) async fn run_run(
    database: &sea_orm::DatabaseConnection,
    args: RunArgs,
) -> Result<(), DbErr> {
    let run_id = match queued_run_id(&args.targets)? {
        Some(run_id) => run_id,
        None => {
            let resolved = resolve_targets(&args.targets)?;
            let backend = args.backend.map_or_else(default_backend, Ok)?;
            let run = match backend {
                Backend::Openfold => {
                    submit_openfold_run(database, &args, &resolved, &batch_inputs_dir()).await?
                }
                Backend::Esmfold => submit_esmfold_run(database, &args, &resolved).await?,
            };
            if !args.json {
                println!(
                    "Queued {} run {} ({}, {} residues)\n",
                    backend.label(),
                    run.id,
                    run.input_id,
                    resolved
                        .iter()
                        .map(|target| target.example.residues)
                        .sum::<usize>()
                );
            }
            if args.no_exec {
                if args.json {
                    println!("{}", json!({ "run_id": run.id, "status": run.status }));
                }
                return Ok(());
            }
            run.id
        }
    };

    if args.json {
        execute_run(database, run_id, &LocalCommandRunner).await?;
    } else {
        report_execution(database, run_id).await?;
        println!();
    }
    run_artifacts::register_run_artifacts(database, run_id).await?;

    let run = runs::get_run_with_artifacts(database, run_id)
        .await?
        .map(|result| result.run)
        .ok_or_else(|| DbErr::Custom(format!("run {run_id} does not exist")))?;
    let elapsed = run
        .started_at
        .zip(run.completed_at)
        .map(|(started, completed)| (completed - started).num_seconds());

    if args.json {
        println!(
            "{}",
            json!({ "run_id": run.id, "status": run.status, "elapsed_s": elapsed })
        );
    } else if run.status == "completed" {
        let took = elapsed.map_or(String::new(), |seconds| format!(" in {seconds}s"));
        println!(
            "Run {} completed{took}. View it with: vizfold serve",
            run.id
        );
    }

    // A failed fold must exit non-zero: a `set -e` script or SLURM step has nothing else to test.
    if run.status == "completed" {
        return Ok(());
    }
    Err(DbErr::Custom(format!(
        "run {} finished with status: {}{}",
        run.id,
        run.status,
        run.error_message
            .map(|message| format!("\n{message}"))
            .unwrap_or_default()
    )))
}

pub(super) fn unknown_target(target: &str) -> DbErr {
    let available: Vec<String> = examples::scan_default()
        .into_iter()
        .map(|example| example.id)
        .collect();
    DbErr::Custom(if available.is_empty() {
        format!(
            "'{target}' is not a run id, and no bundled examples were found under {}; re-run `vizfold install openfold`",
            examples::monomer_dir().display()
        )
    } else {
        format!(
            "'{target}' is not a run id or a bundled example; available examples: {}",
            available.join(", ")
        )
    })
}

/// A lone integer target replays that run; mixed with anything else it is ambiguous.
pub(super) fn queued_run_id(targets: &[String]) -> Result<Option<i32>, DbErr> {
    let ids: Vec<i32> = targets
        .iter()
        .filter_map(|target| target.parse().ok())
        .collect();
    match (ids.first(), targets.len()) {
        (None, _) => Ok(None),
        (Some(&run_id), 1) => Ok(Some(run_id)),
        (Some(_), _) => Err(DbErr::Custom(
            "a queued run id folds on its own; it cannot be mixed with other targets".to_owned(),
        )),
    }
}

#[derive(Debug)]
pub(super) struct Target {
    fasta: PathBuf,
    example: examples::Example,
}

pub(super) fn tags(resolved: &[Target]) -> Vec<&str> {
    resolved
        .iter()
        .map(|target| target.example.id.as_str())
        .collect()
}

/// Every FASTA the targets name, in order: an example id, a path, or a directory of FASTAs.
pub(super) fn resolve_targets(targets: &[String]) -> Result<Vec<Target>, DbErr> {
    let mut resolved: Vec<Target> = Vec::new();
    for target in targets {
        let path = local_path(target);
        // `read_fasta` below re-parses each file and cannot see the alignments directory beside
        // it, so carry the scan's answer down. Only a bundled protein can have one.
        let (fastas, aligned) = match examples::find(target) {
            // The example's file, not its directory: a stray sibling FASTA cannot join the fold.
            Some(example) => (
                examples::first_fasta(Path::new(&default_fasta_dir(&example.id)))
                    .into_iter()
                    .collect(),
                example.alignments,
            ),
            None if path.is_dir() => (examples::fasta_files(&path), false),
            // Anything path-shaped falls through to `read_fasta`, which names what it could not read.
            None if path.extension().is_some() || path.exists() => (vec![path], false),
            None => return Err(unknown_target(target)),
        };
        if fastas.is_empty() {
            return Err(DbErr::Custom(format!(
                "'{target}' holds no .fasta or .fa files"
            )));
        }
        for fasta in fastas {
            let example = examples::Example {
                alignments: aligned,
                ..read_fasta(&fasta)?
            };
            // The tag becomes a directory and a link name here; preflight takes nothing else either.
            if !example
                .id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                return Err(DbErr::Custom(format!(
                    "'{}' is tagged '{}'; an OpenFold tag is letters, digits and underscores",
                    fasta.display(),
                    example.id
                )));
            }
            // Every OpenFold output is keyed by tag, so a repeat would overwrite the first.
            if resolved.iter().any(|other| other.example.id == example.id) {
                return Err(DbErr::Custom(format!(
                    "'{}' is folded twice by these targets",
                    example.id
                )));
            }
            let fasta = std::fs::canonicalize(&fasta).map_err(|error| {
                DbErr::Custom(format!("cannot resolve '{}': {error}", fasta.display()))
            })?;
            resolved.push(Target { fasta, example });
        }
    }
    Ok(resolved)
}

/// Where staged batches live: outside the checkout, beside the run outputs the workbench serves.
pub(super) fn batch_inputs_dir() -> PathBuf {
    config::prefix().join("runs/inputs")
}

/// cwd first, then `base`: a relative path means what the shell means by it, and an absolute one
/// resolves to itself because `cwd.join(absolute)` is that absolute path.
fn beside_cwd_or(path: &Path, base: &Path) -> PathBuf {
    std::env::current_dir()
        .ok()
        .map(|cwd| cwd.join(path))
        .filter(|candidate| candidate.exists())
        .unwrap_or_else(|| base.join(path))
}

/// cwd first, then the checkout: a relative target means what the shell means by it.
pub(super) fn local_path(target: &str) -> PathBuf {
    beside_cwd_or(Path::new(target), &config::openfold_home())
}

/// One directory of symlinks, so OpenFold's single `fasta_dir` can name a whole batch. Each link
/// is replaced in place rather than the directory wiped, which would empty a concurrent run of it.
pub(super) fn stage_batch(inputs_dir: &Path, resolved: &[Target]) -> Result<PathBuf, DbErr> {
    let dir = inputs_dir.join(batch_name(resolved));
    std::fs::create_dir_all(&dir).map_err(|error| {
        DbErr::Custom(format!(
            "cannot stage the batch at {}: {error}",
            dir.display()
        ))
    })?;
    for target in resolved {
        let link = dir.join(format!("{}.fasta", target.example.id));
        std::fs::remove_file(&link).ok();
        std::os::unix::fs::symlink(&target.fasta, &link).map_err(|error| {
            DbErr::Custom(format!(
                "cannot link {} into {}: {error}",
                target.fasta.display(),
                dir.display()
            ))
        })?;
    }
    Ok(dir)
}

/// `<tags>-<hash of the files behind them>`: same batch, same directory; same tags over different
/// files, a different one -- so a submit cannot relink a batch already recorded and waiting to fold.
/// Long tag lists collapse to first+count.
pub(super) fn batch_name(resolved: &[Target]) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for target in resolved {
        target.fasta.hash(&mut hasher);
    }
    let joined = tags(resolved).join("+");
    let head = if joined.len() <= 80 {
        joined
    } else {
        format!("{}+{}more", resolved[0].example.id, resolved.len() - 1)
    };
    format!("{head}-{:x}", hasher.finish())
}

/// The seeded records a local run is built from; both backends differ only in which they name.
pub(super) struct LocalCatalog {
    backend_id: i32,
    target_id: i32,
    profile_id: i32,
    available_resources_json: String,
    provenance: String,
    working_dir: String,
}

pub(super) async fn local_catalog(
    database: &sea_orm::DatabaseConnection,
    backend: Backend,
) -> Result<LocalCatalog, DbErr> {
    // Submitting needs the catalog, so seed here rather than per caller. Guarded, so repeating is free.
    seed_defaults(database).await?;
    let model = model_backend_entity::Entity::find()
        .filter(model_backend_entity::Column::Slug.eq(backend.slug()))
        .one(database)
        .await?
        .ok_or_else(seed_required_error)?;
    let target = execution_target_entity::Entity::find()
        .filter(execution_target_entity::Column::Slug.eq(format!("local-{}", backend.slug())))
        .one(database)
        .await?
        .ok_or_else(seed_required_error)?;
    let profile = model_invocation_profiles::list_model_invocation_profiles(database)
        .await?
        .into_iter()
        .find(|profile| {
            profile.model_backend_id == model.id
                && profile.execution_target_id == target.id
                && profile.invocation_kind == "local_subprocess"
        })
        .ok_or_else(seed_required_error)?;
    Ok(LocalCatalog {
        backend_id: model.id,
        target_id: target.id,
        profile_id: profile.id,
        available_resources_json: target.available_resources_json,
        provenance: runs::provenance_snapshot(
            &model.slug,
            model.version.as_deref(),
            &target.slug,
            &profile.invocation_kind,
            &profile.config_json,
            &config::openfold_home(),
            &config::prefix(),
            &backend.env_prefix(),
        ),
        working_dir: local_working_dir(&profile)?,
    })
}

pub(super) async fn submit_openfold_run(
    database: &sea_orm::DatabaseConnection,
    args: &RunArgs,
    resolved: &[Target],
    inputs_dir: &Path,
) -> Result<crate::core::entities::runs::Model, DbErr> {
    let catalog = local_catalog(database, Backend::Openfold).await?;
    let working_dir = &catalog.working_dir;
    // One target is its own `fasta_dir`; more than one needs a directory that names them all.
    let fasta_dir = match resolved {
        [only] => only.fasta.clone(),
        many => stage_batch(inputs_dir, many)?,
    };
    let data_dir_input = args
        .data_dir
        .clone()
        .unwrap_or_else(|| config::data_dir().to_string_lossy().into_owned());
    let data_dir = canonicalize_local_path("--data-dir", &data_dir_input, working_dir)?;
    // Keyed on the alignments each target actually has, not on being bundled: a bundled protein
    // with no `alignments/<id>` would otherwise borrow the directory and fold against the wrong MSA.
    let use_precomputed_alignments = args
        .use_precomputed_alignments
        .unwrap_or_else(|| resolved.iter().all(|target| target.example.alignments));
    let alignment_dir = if use_precomputed_alignments {
        let input = args
            .alignment_dir
            .clone()
            .unwrap_or_else(default_alignment_dir);
        Some(canonicalize_local_path(
            "--alignment-dir",
            &input,
            working_dir,
        )?)
    } else {
        args.alignment_dir
            .as_deref()
            .map(|path| canonicalize_local_path("--alignment-dir", path, working_dir))
            .transpose()?
    };

    let model_device = args
        .model_device
        .clone()
        .unwrap_or_else(default_model_device);

    let mut execution_parameters = serde_json::Map::from_iter([
        ("fasta_dir".into(), json!(fasta_dir)),
        ("data_dir".into(), json!(data_dir)),
        ("residue_idx".into(), json!(args.residue_idx)),
        (
            "use_precomputed_alignments".into(),
            json!(use_precomputed_alignments),
        ),
        ("model_device".into(), json!(model_device)),
        (
            "cpus".into(),
            json!(clamp_cpus(
                args.cpus.unwrap_or_else(default_cpus),
                &catalog.available_resources_json
            )),
        ),
    ]);
    if let Some(alignment_dir) = alignment_dir {
        execution_parameters.insert("alignment_dir".into(), json!(alignment_dir));
    }

    let run = runs::submit_run(
        database,
        runs::SubmitRunInput {
            model_backend_id: catalog.backend_id,
            execution_target_id: catalog.target_id,
            invocation_profile_id: catalog.profile_id,
            status: "submitted".into(),
            // ponytail: `input_id.split('+')` is how preflight learns what the batch must hold.
            input_id: args
                .input_id
                .clone()
                .unwrap_or_else(|| tags(resolved).join("+")),
            input_sequence: resolved
                .iter()
                .map(|target| target.example.sequence.as_str())
                .collect::<Vec<_>>()
                .join(":"),
            // demo_attn on the wire: the Python argument name and the seed schema key.
            model_parameters_json: json!({
                "save_outputs": args.save_outputs,
                "demo_attn": args.attn,
                "num_recycles_save": args.num_recycles_save,
            })
            .to_string(),
            execution_parameters_json: serde_json::Value::Object(execution_parameters).to_string(),
            provenance_json: Some(catalog.provenance),
        },
    )
    .await?;

    Ok(run)
}

/// ESMFold reads one file and loads its model inside the fold, so a batch has nowhere to go.
pub(super) async fn submit_esmfold_run(
    database: &sea_orm::DatabaseConnection,
    args: &RunArgs,
    resolved: &[Target],
) -> Result<crate::core::entities::runs::Model, DbErr> {
    let [target] = resolved else {
        return Err(DbErr::Custom(
            "ESMFold folds one target at a time; pass a single FASTA.".to_owned(),
        ));
    };
    let catalog = local_catalog(database, Backend::Esmfold).await?;
    let model_device = args
        .model_device
        .clone()
        .unwrap_or_else(default_model_device);

    let run = runs::submit_run(
        database,
        runs::SubmitRunInput {
            model_backend_id: catalog.backend_id,
            execution_target_id: catalog.target_id,
            invocation_profile_id: catalog.profile_id,
            status: "submitted".into(),
            input_id: args
                .input_id
                .clone()
                .unwrap_or_else(|| target.example.id.clone()),
            input_sequence: target.example.sequence.clone(),
            model_parameters_json: json!({
                "model": args.model,
                "trace_mode": args.trace_mode,
                "layers": args.layers,
                "dtype": args.dtype,
                "save_fp16": args.save_fp16,
                "structure_traces": args.structure_traces,
            })
            .to_string(),
            execution_parameters_json: json!({
                "fasta": target.fasta,
                "model_device": model_device,
            })
            .to_string(),
            provenance_json: Some(catalog.provenance),
        },
    )
    .await?;

    Ok(run)
}

pub(super) fn default_backend() -> Result<Backend, DbErr> {
    match (
        Backend::Openfold.is_installed(),
        Backend::Esmfold.is_installed(),
    ) {
        (false, true) => Ok(Backend::Esmfold),
        (true, _) => Ok(Backend::Openfold),
        (false, false) => Err(DbErr::Custom(
            "no backend is installed; run `vizfold install openfold` or `vizfold install esmfold`"
                .to_owned(),
        )),
    }
}

/// `<OPENFOLD_HOME>/examples/monomer/fasta_dir_<id-stem>` -- the id up to its last underscore.
pub(super) fn default_fasta_dir(input_id: &str) -> String {
    let stem = input_id.rsplit_once('_').map_or(input_id, |(head, _)| head);
    config::openfold_home()
        .join("examples/monomer")
        .join(format!("fasta_dir_{stem}"))
        .to_string_lossy()
        .into_owned()
}

pub(super) fn default_alignment_dir() -> String {
    config::openfold_home()
        .join("examples/monomer/alignments")
        .to_string_lossy()
        .into_owned()
}

pub(super) fn local_working_dir(
    profile: &crate::core::entities::model_invocation_profiles::Model,
) -> Result<String, DbErr> {
    let config: serde_json::Value =
        serde_json::from_str(&profile.config_json).map_err(|error| {
            DbErr::Custom(format!(
                "local invocation profile config_json must be valid JSON: {error}"
            ))
        })?;
    config
        .get("working_dir")
        .and_then(serde_json::Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            DbErr::Custom(
                "local invocation profile config_json requires a non-empty working_dir".into(),
            )
        })
}

/// cwd first, then the target's working dir: a relative path means what the shell means by it.
pub(super) fn canonicalize_local_path(
    field: &str,
    path: &str,
    working_dir: &str,
) -> Result<String, DbErr> {
    let resolved = beside_cwd_or(Path::new(path), Path::new(working_dir));
    canonicalize_at(field, path, &resolved)
}

pub(super) fn canonicalize_at(
    field: &str,
    original: &str,
    attempted: &Path,
) -> Result<String, DbErr> {
    std::fs::canonicalize(attempted)
        .map(|path| path.display().to_string())
        .map_err(|error| {
            DbErr::Custom(format!(
                "{field} original path '{original}' could not be resolved at '{}': {error}",
                attempted.display()
            ))
        })
}

/// The one sequence at a resolved FASTA path, as the repo of a run's id and sequence.
pub(super) fn read_fasta(path: &Path) -> Result<examples::Example, DbErr> {
    examples::from_path(path).ok_or_else(|| {
        DbErr::Custom(format!(
            "no FASTA record at '{}': expected a .fasta/.fa file with a '>' header and a sequence",
            path.display()
        ))
    })
}

/// Seeding runs immediately before every lookup, so a miss means the database is not one vizfold wrote.
pub(super) fn seed_required_error() -> DbErr {
    DbErr::Custom(format!(
        "the run's backend, local execution target, or matching profile is missing from {} \
         even after seeding; point VIZFOLD_DB at a vizfold database, or remove that file",
        config::database_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(config::database_url)
    ))
}

#[cfg(test)]
mod tests {
    use super::super::test_support::run_args;
    use super::*;

    fn fake_target(id: &str, fasta: &str) -> Target {
        Target {
            fasta: PathBuf::from(fasta),
            example: examples::Example {
                id: id.to_owned(),
                residues: 1,
                description: String::new(),
                sequence: "M".to_owned(),
                alignments: false,
            },
        }
    }

    /// Two tags name the directory; twenty collapse to a stable short name.
    #[test]
    fn a_batch_directory_is_named_after_its_tags_and_its_files() {
        let pair = [
            fake_target("1UBQ_1", "/a/1UBQ.fasta"),
            fake_target("6KWC_1", "/a/6KWC.fasta"),
        ];
        let name = super::batch_name(&pair);
        assert!(name.starts_with("1UBQ_1+6KWC_1-"), "{name}");
        assert_eq!(
            name,
            super::batch_name(&pair),
            "the same batch, the same directory"
        );

        // Same tags, other files: a recorded batch cannot be relinked by a later submit.
        let elsewhere = [
            fake_target("1UBQ_1", "/b/1UBQ.fasta"),
            fake_target("6KWC_1", "/a/6KWC.fasta"),
        ];
        assert_ne!(name, super::batch_name(&elsewhere));

        let many: Vec<Target> = (0..20)
            .map(|index| fake_target("1UBQ_1", &format!("/a/{index}.fasta")))
            .collect();
        let long = super::batch_name(&many);
        assert!(long.len() < 40, "{long} is too long for a directory name");
        assert!(long.starts_with("1UBQ_1+19more-"), "{long}");
    }

    // The guard is held across awaits on purpose: `#[tokio::test]` gives each test its own
    // current-thread runtime driving one future, so nothing inside this runtime can want the lock,
    // and the contention it exists to serialise is between test threads.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn submit_openfold_run_uses_seeded_records() -> Result<(), DbErr> {
        let _env = crate::core::test_support::env_lock();
        let local_path = std::fs::canonicalize(crate::core::config::openfold_home())
            .expect("OpenFold home should be canonicalizable")
            .display()
            .to_string();
        let database = crate::core::test_support::seeded_db().await?;

        // A real FASTA, so this pins the id and sequence derivation too.
        let fasta = std::fs::canonicalize(
            crate::core::examples::monomer_dir().join("fasta_dir_6KWC/6KWC.fasta"),
        )
        .expect("the bundled 6KWC example should exist")
        .display()
        .to_string();

        let args = run_args(&[
            &fasta,
            "--data-dir",
            &local_path,
            "--alignment-dir",
            &local_path,
            "--model-device",
            "cpu",
            // Over the seeded target's cpus.maximum of 14, so the run must record the clamped value.
            "--cpus",
            "18",
            "--use-precomputed-alignments=true",
        ]);
        submit_openfold_run(
            &database,
            &args,
            &super::resolve_targets(&args.targets)?,
            &super::batch_inputs_dir(),
        )
        .await?;

        let runs = runs::list_runs(&database).await?;
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].status, "submitted");
        assert_eq!(
            runs[0].input_id, "6KWC_1",
            "the id comes from the FASTA header"
        );
        assert_eq!(
            runs[0].input_sequence.len(),
            191,
            "the sequence comes from the FASTA"
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&runs[0].model_parameters_json)
                .expect("model parameters should be valid JSON"),
            json!({"save_outputs": true, "demo_attn": true, "num_recycles_save": 1})
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&runs[0].execution_parameters_json)
                .expect("execution parameters should be valid JSON"),
            // The file itself, not the directory holding it.
            json!({"fasta_dir": fasta, "data_dir": local_path, "alignment_dir": local_path, "residue_idx": 1, "use_precomputed_alignments": true, "model_device": "cpu", "cpus": 14})
        );

        let provenance: serde_json::Value = serde_json::from_str(
            runs[0]
                .provenance_json
                .as_deref()
                .expect("provenance_json should be set"),
        )
        .expect("provenance_json should be valid JSON");
        assert_eq!(
            provenance["profile"]["config"]["output_location"],
            json!(config::prefix().join("runs"))
        );
        Ok(())
    }

    /// Regression: a bundled example names a directory, and ESMFold's `--fasta` needs the file in it.
    #[test]
    fn an_example_resolves_to_its_fasta_file_not_its_directory() {
        let _env = crate::core::test_support::env_lock();
        let resolved = super::resolve_targets(&["6KWC_1".to_owned()]).expect("6KWC_1 resolves");
        let [target] = resolved.as_slice() else {
            panic!("one target, got {}", resolved.len());
        };
        assert!(
            std::path::Path::new(&target.fasta).is_file(),
            "resolved '{}' should be the file, not the directory",
            target.fasta.display()
        );
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn two_targets_submit_one_run_over_a_staged_directory() -> Result<(), DbErr> {
        let _env = crate::core::test_support::env_lock();
        let database = crate::core::test_support::seeded_db().await?;
        let home = config::openfold_home().display().to_string();
        let args = run_args(&["1UBQ_1", "6KWC_1", "--data-dir", &home]);
        let resolved = super::resolve_targets(&args.targets)?;

        assert!(
            resolved.iter().all(|target| target.example.alignments),
            "bundled proteins default to their precomputed alignments"
        );
        // Not the real prefix: a submit host may have it read-only, and this test writes there.
        let inputs = std::env::temp_dir().join(format!("vizfold-batch-{}", std::process::id()));

        submit_openfold_run(&database, &args, &resolved, &inputs).await?;

        let runs = runs::list_runs(&database).await?;
        assert_eq!(runs.len(), 1, "one row per invocation");
        assert_eq!(runs[0].input_id, "1UBQ_1+6KWC_1");
        let execution: serde_json::Value =
            serde_json::from_str(&runs[0].execution_parameters_json).expect("valid JSON");
        assert_eq!(execution["use_precomputed_alignments"], json!(true));

        let staged = PathBuf::from(execution["fasta_dir"].as_str().expect("a fasta_dir"));
        assert_eq!(staged.parent(), Some(inputs.as_path()));
        assert!(
            staged
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("1UBQ_1+6KWC_1-")),
            "{}",
            staged.display()
        );
        let mut staged_names: Vec<String> = std::fs::read_dir(&staged)
            .expect("the batch should be staged")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        staged_names.sort();
        assert_eq!(staged_names, ["1UBQ_1.fasta", "6KWC_1.fasta"]);
        // Links, not copies, so the batch directory costs nothing and cannot drift from its repo.
        assert!(staged.join("1UBQ_1.fasta").is_file());
        std::fs::remove_dir_all(&inputs).ok();
        Ok(())
    }

    #[tokio::test]
    async fn esmfold_refuses_more_than_one_target() -> Result<(), DbErr> {
        let database = crate::core::test_support::seeded_db().await?;
        let args = run_args(&["1UBQ_1", "6KWC_1", "--backend", "esmfold"]);
        let resolved = super::resolve_targets(&args.targets)?;

        let error = submit_esmfold_run(&database, &args, &resolved)
            .await
            .expect_err("a batch should be refused");

        assert!(error.to_string().contains("one target at a time"));
        Ok(())
    }

    /// A repeated tag is refused, not overwritten.
    #[test]
    fn a_directory_target_resolves_to_every_fasta_in_it() {
        let dir = crate::core::examples::monomer_dir().join("fasta_dir_6KWC");
        let resolved = super::resolve_targets(&[dir.display().to_string()]).expect("6KWC exists");
        assert_eq!(super::tags(&resolved), ["6KWC_1"]);
        assert!(
            !resolved[0].example.alignments,
            "a path is the user's own, so it borrows no alignments"
        );

        let twice = super::resolve_targets(&["6KWC_1".into(), dir.display().to_string()]);
        assert!(
            twice.is_err_and(|error| error.to_string().contains("folded twice")),
            "one tag cannot appear twice in a batch"
        );
    }

    /// A tag names a staged directory and a link, so a path in a FASTA header must not reach the disk.
    #[test]
    fn a_header_that_is_not_a_tag_is_refused() {
        let dir = std::env::temp_dir().join(format!("vizfold-tag-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("fixture");
        let fasta = dir.join("evil.fasta");
        std::fs::write(&fasta, ">../../../etc/passwd\nMQIFVKTL\n").expect("fixture");

        let error = super::resolve_targets(&[fasta.display().to_string()])
            .expect_err("a traversal tag should be refused");

        assert!(
            error
                .to_string()
                .contains("letters, digits and underscores"),
            "{error}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn submit_openfold_run_reports_missing_local_path() -> Result<(), DbErr> {
        let _env = crate::core::test_support::env_lock();
        let database = crate::core::test_support::seeded_db().await?;
        let missing_path = "definitely-missing-vizfold-local-path";
        let args = run_args(&["6KWC_1", "--data-dir", missing_path]);

        let error = submit_openfold_run(
            &database,
            &args,
            &super::resolve_targets(&args.targets)?,
            &super::batch_inputs_dir(),
        )
        .await
        .expect_err("missing local path should fail");

        assert!(error.to_string().contains(
            "--data-dir original path 'definitely-missing-vizfold-local-path' could not be resolved"
        ));
        assert!(
            error
                .to_string()
                .contains(&crate::core::config::openfold_home().display().to_string())
        );
        Ok(())
    }

    #[test]
    fn an_unknown_target_is_refused_before_anything_is_submitted() {
        assert!(super::resolve_targets(&["not-a-thing".into()]).is_err());
        assert!(
            super::resolve_targets(&["./typo.fasta".into()])
                .is_err_and(|error| error.to_string().contains("no FASTA record at")),
            "a path-shaped target is reported as a path"
        );
    }

    #[test]
    fn model_device_workstation_defaults_to_cpu_without_a_gpu() {
        assert_eq!(
            super::model_device_for(config::SlurmContext::None, None, None),
            "cpu"
        );
    }

    #[test]
    fn model_device_workstation_defaults_to_cuda_with_a_gpu() {
        assert_eq!(
            super::model_device_for(config::SlurmContext::None, None, Some("NVIDIA A100")),
            "cuda:0"
        );
    }

    #[test]
    fn model_device_prefers_the_configured_gpu_partition_without_probing() {
        assert_eq!(
            super::model_device_for(config::SlurmContext::None, Some("gpuA100x4"), None),
            "cuda:0"
        );
    }

    #[test]
    fn model_device_inside_an_allocation_trusts_the_local_probe() {
        assert_eq!(
            super::model_device_for(config::SlurmContext::InAllocation, Some("gpuA100x4"), None),
            "cpu"
        );
    }

    /// An absent or unparseable maximum must not clamp the request to something arbitrary.
    #[test]
    fn cpus_clamp_only_where_the_target_declares_a_maximum() {
        let with_max = json!({"properties": {"cpus": {"maximum": 14}}}).to_string();
        assert_eq!(super::clamp_cpus(18, &with_max), 14);
        assert_eq!(super::clamp_cpus(8, &with_max), 8);
        assert_eq!(
            super::clamp_cpus(18, &json!({"properties": {}}).to_string()),
            18
        );
        assert_eq!(super::clamp_cpus(18, "not-json"), 18);
    }

    #[test]
    fn a_run_id_refuses_to_share_the_command_line() {
        let ids = |argv: &[&str]| super::queued_run_id(&run_args(argv).targets);
        assert_eq!(ids(&["42"]).expect("a lone id replays"), Some(42));
        assert_eq!(ids(&["1UBQ_1"]).expect("no id here"), None);
        assert_eq!(ids(&["1UBQ_1", "6KWC_1"]).expect("no id here"), None);
        assert!(ids(&["42", "6KWC_1"]).is_err());
    }
}
