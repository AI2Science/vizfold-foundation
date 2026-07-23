use executor::core::commands::{CommandRunner, CommandSpec, LocalCommandRunner};

#[tokio::main]
async fn main() -> Result<(), sea_orm::DbErr> {
    let spec = demo_command();
    let output = LocalCommandRunner.run(spec).await?;

    println!("exit_code: {}", output.exit_code);
    println!("stdout:\n{}", output.stdout);
    println!("stderr:\n{}", output.stderr);

    Ok(())
}

#[cfg(unix)]
fn demo_command() -> CommandSpec {
    CommandSpec {
        program: "sh".into(),
        args: vec!["-c".into(), "printf 'local command runner demo\\n'".into()],
        ..Default::default()
    }
}

#[cfg(windows)]
fn demo_command() -> CommandSpec {
    CommandSpec {
        program: "cmd".into(),
        args: vec!["/C".into(), "echo local command runner demo".into()],
        ..Default::default()
    }
}
