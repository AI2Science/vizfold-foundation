//! Helpers shared by more than one module's tests.

use clap::Parser;

use super::args::{Cli, Command, RunArgs};

pub(super) fn run_args(argv: &[&str]) -> RunArgs {
    let full: Vec<&str> = ["vizfold", "run"]
        .into_iter()
        .chain(argv.iter().copied())
        .collect();
    match Cli::try_parse_from(full)
        .expect("run argv should parse")
        .command
    {
        Command::Run(args) => args,
        other => panic!("not a run: {other:?}"),
    }
}
